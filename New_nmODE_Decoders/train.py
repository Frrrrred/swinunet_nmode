import sys
import warnings
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter

from models.unet import UNet
from models.unet_EED import UNet_EED
from models.unet_LMD import UNet_LMD
from models.unet_HD import UNet_HD
from models.unet_FED import UNet_FED
from models.unet_EAD import UNet_EAD
from models.unet_RKD import UNet_RKD
from engine import *
from utils import *
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")
torch.cuda.set_device(0)


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')

    if not config.work_dir:
        raise ValueError("config.work_dir is not set")

    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    if config.network == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif config.network == 'unet_eed':
        model = UNet_EED(activefunc='gelu',droprate=0,kernel_size=3,n_channels=3, n_classes=1)
    elif config.network == 'unet_lmd':
        model = UNet_LMD(activefunc='gelu',droprate=0,kernel_size=3,n_channels=3, n_classes=1)
    elif config.network == 'unet_hd':
        model = UNet_HD(activefunc='gelu',droprate=0,kernel_size=3,n_channels=3, n_classes=1)
    # 新增模型
    elif config.network == 'unet_fed':
        model = UNet_FED(activefunc='gelu',droprate=0.05,kernel_size=3,n_channels=3, n_classes=1)
    elif config.network == 'unet_ead':
        model = UNet_EAD(activefunc='gelu',droprate=0.05,kernel_size=3,n_channels=3, n_classes=1)
    elif config.network == 'unet_rkd':
        model = UNet_RKD(activefunc='gelu',droprate=0.05,kernel_size=3,n_channels=3, n_classes=1)
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    delta_lr = 0.1
    optimizer = get_optimizer(config, model,config.lr)
    # 使用 ASGD 优化 delta
    delta_optimizer = torch.optim.ASGD([model.delta], lr=delta_lr)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = (f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f},'
                    f' min_epoch: {min_epoch}, loss: {loss:.4f}')
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            delta_optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                delta_optimizer,
                epoch,
                logger,
                config
        )

        if loss < min_loss:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'delta': model.delta,
                 }, os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'delta': model.delta,
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        best_weight = best['model_state_dict']
        best_delta = best['delta']
        model.load_state_dict(best_weight)
        model.delta = best_delta

        loss, miou = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
        )
        print(f'best model epoch: {min_epoch}, test loss: {loss:.4f}, test miou: {miou:.4f}')

        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)
