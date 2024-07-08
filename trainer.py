import torch
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, args, model, train_cfg, current_epoch, device):
        super(Trainer, self).__init__()
        assert current_epoch > 0
        
        self.args =args
        self.model = model
        self.train_cfg = train_cfg

        self.current_epoch = current_epoch
        self.current_phase = None
        self.num_device = 1 if device == 'cpu' else args.gpus

    def test(self):
        # setup dataloader
        self.model.setup('test')
        test_loader = self.model.test_dataloader()

        self.model.eval()
        outputs = []
        for batch in tqdm(test_loader):
            logs = self.model.test_step(batch)
            outputs.append(logs)

        self.model.test_epoch_end(outputs)