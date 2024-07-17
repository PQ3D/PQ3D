import contextlib
import copy
import numpy as np
from tqdm import tqdm
from time import time

import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable the warning of tokenizers parallelism


@TRAINER_REGISTRY.register()
class MultitaskTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # convert loaders and evaluators to list
        for split in ['val', 'test']:
            if split not in self.data_loaders:
                continue
            loaders = self.data_loaders[split]
            if not (isinstance(loaders, list) or isinstance(loaders, tuple)):
                self.data_loaders[split] = [loaders]
        if not isinstance(self.evaluator, list):
            self.evaluator = [self.evaluator]
            
        if cfg.get('profile'):
            def trace_handler(p):
                tb_handler = torch.profiler.tensorboard_trace_handler('./outputs/profile_trace/')
                tb_handler(p)
                output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total")
                print(output)

            profile = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=10, warmup=10, active=10, repeat=1),
                    on_trace_ready=trace_handler,
                    profile_memory=True, record_shapes=True, with_modules=True, with_stack=True)
            profile.start()
            self.profile = profile
        else:
            self.profile = contextlib.nullcontext()

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        for i, data_dict in enumerate(loader):
            if isinstance(self.profile, torch.profiler.profile):
                self.profile.step()
            with self.accelerator.accumulate(self.model):
                data_dict['cur_step'] = epoch * len(loader) + i
                data_dict['total_steps'] = self.total_steps
                data_dict = self.forward(data_dict)
                loss, losses = self.loss(data_dict)
                self.backward(loss)
                # record
                self.global_step += 1
                log_dict = {'step': self.global_step}
                log_dict.update(losses)
                self.log(log_dict, mode="train")
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        target_metrics = []
        loaders = self.data_loaders["val"]
        evaluators = self.evaluator
        for loader, evaluator in zip(loaders, evaluators):
            st = time()
            is_best, results = self.eval(loader, evaluator)
            end = time()
            results['time'] = (end - st) / 60
            self.log(results, mode="val-" + loader.dataset.dataset.__class__.__name__) 
            evaluator.reset()
            target_metrics.append(results['target_metric'])
        target_metric = np.sum(target_metrics)
        if target_metric > self.exp_tracker.best_result:
            is_best = True
            self.exp_tracker.best_result = target_metric
        else:
            is_best = False
        return is_best

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loaders = self.data_loaders["test"]
        evaluators = self.evaluator
        for loader, evaluator in zip(loaders, evaluators):
            is_best, results = self.eval(loader, evaluator)
            self.log(results, mode= "test-" + loader.dataset.dataset.__class__.__name__)
            evaluator.reset()
        return results
    
    @torch.no_grad()
    def eval(self, loader, evaluator):
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=loader.dataset.dataset.__class__.__name__)
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            self.postprocess_for_eval(data_dict, loader)
            evaluator.update(data_dict)
            pbar.update(1)
        return evaluator.record()

    def run(self):
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                st = time()
                self.train_step(epoch)
                epoch_time = (time() - st) / 60
                self.log({"epoch_time": epoch_time, "epoch": epoch+1, "remaining_time": epoch_time * (self.epochs - epoch) / 60}, mode="train")

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")
                    self.save("latest.pth")

        self.test_step()
        if self.mode == "train":
            self.accelerator.end_training()

    def postprocess_for_eval(self, data_dict, loader):
        if data_dict['task_id'][0] in [0]: # 0 for refer task
            return
        if 'generation_logits' in data_dict.keys():
            tokenizer = loader.dataset.generation_tokenizer
            response_pred = tokenizer.batch_decode(data_dict['generation_logits'], skip_special_tokens=True)
            data_dict['caption_pred'] = response_pred
            data_dict['answer_pred'] = response_pred


