import os
import glob
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset 
from utils.eval_model import eval
from utils.vis import image_with_boxes
import numpy as np

def train(model,
          model_type,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):
    best_acc = 0.0
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            images, labels, atts = data
            images, labels, atts = images.cuda(), labels.cuda(), atts.cuda()

            optimizer.zero_grad()

            if model_type == 'kfn':
                logits = model(images, atts)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            else:
                proposalN_windows_score, proposalN_windows_logits, indices, \
                window_scores, coordinates, logits = model(images, atts, epoch, i, 'train')

                loss = criterion(logits, labels)

                windowscls_loss = criterion(proposalN_windows_logits,
                                   labels.unsqueeze(1).repeat(1, proposalN).view(-1))

                if epoch < 2:
                    total_loss = loss
                else:
                    total_loss = loss + windowscls_loss

                total_loss.backward()

                optimizer.step()
                # object branch tensorboard
                if i == 1:
                    with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train' + 'cut') as writer:
                        cat_imgs = []
                        for j, coordinate_ndarray in enumerate(coordinates):
                            img = image_with_boxes(images[j], coordinate_ndarray)
                            cat_imgs.append(img)
                        cat_imgs = np.concatenate(cat_imgs, axis=1)
                        writer.add_images('train' + '/' + 'cut image with windows', cat_imgs, epoch, dataformats='HWC')

        scheduler.step()

        # evaluation every epoch
        if eval_trainset:
            loss_avg, windowscls_loss_avg, total_loss_avg, accuracy = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: accuracy: {:.2f}%'.format(100. * accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:

                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/accuracy', accuracy, epoch)
                writer.add_scalar('Train/loss_avg', loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval testset
        if model_type == 'kfn':
            loss_avg, accuracy = eval(model, model_type, testloader, criterion, 'test', save_path, epoch)
            if accuracy > best_acc:
                best_acc = accuracy
                print('Saving the best checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                }, os.path.join(save_path, 'best', 'best' + '.pth'))

            print(
                'Test set: accuracy: {:.2f}%, best accuracy {:.2f}%'.format(
                    100. * accuracy, 100. * best_acc))
            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
                writer.add_scalar('Test/accuracy', accuracy, epoch)
                writer.add_scalar('Test/loss_avg', loss_avg, epoch)

        else:
            loss_avg, windowscls_loss_avg, total_loss_avg, accuracy = eval(model, model_type, testloader, criterion, 'test', save_path, epoch)

            if accuracy > best_acc:
                best_acc = accuracy
                print('Saving the best checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                }, os.path.join(save_path, 'best', 'best' + '.pth'))

            print(
                'Test set: accuracy: {:.2f}%, best accuracy {:.2f}%'.format(100. * accuracy, 100. * best_acc))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
                writer.add_scalar('Test/accuracy', accuracy, epoch)
                writer.add_scalar('Test/loss_avg', loss_avg, epoch)
                writer.add_scalar('Test/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))
    print('Best acc:{:.4f}'.format(best_acc))

