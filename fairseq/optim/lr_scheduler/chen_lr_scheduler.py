from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('chen')
class ChenSchedule(FairseqLRScheduler):
    """
    TODO
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.replicas_num = args.replicas_num
        self.warmup_steps = args.warmup_steps
        self.start_step = args.start_step
        self.end_step = args.end_step

        # initial learning rate
        self.lr_init = args.lr[0]
        self.lr = self.lr_init
        self.optimizer.set_lr(self.lr)

        self.step_fn = lambda t: self.lr_init * min(
            1 + t * (self.replicas_num - 1) / (self.replicas_num * self.warmup_steps),
            self.replicas_num,
            self.replicas_num * (
                (self.replicas_num * 2) ** ((self.start_step - self.replicas_num * t) / (self.end_step - self.start_step))))

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--replicas-num', type=int)
        parser.add_argument('--warmup-steps', type=int)
        parser.add_argument('--start-step', type=int)
        parser.add_argument('--end-step', type=int)
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        self.lr = self.step_fn(num_updates)
        self.optimizer.set_lr(self.lr)
        return self.lr
