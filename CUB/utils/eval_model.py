import torch
from tqdm import tqdm
from config import proposalN


def eval(model, model_type, testloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            images, labels, atts = data
            images = images.cuda()
            labels = labels.cuda()
            atts = atts.cuda()

            if model_type == 'kfn':
                logits = model(images, atts)
                loss = criterion(logits, labels)
                loss_sum += loss.item()
                pred = logits.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
            else:
                proposalN_windows_score, proposalN_windows_logits, indices, \
                window_scores, coordinates, logits = model(images, atts, epoch, i, status)

                loss = criterion(logits, labels)
                windowscls_loss = criterion(proposalN_windows_logits,
                                            labels.unsqueeze(1).repeat(1, proposalN).view(-1))

                total_loss = loss +  windowscls_loss

                loss_sum += loss.item()
                windowscls_loss_sum += windowscls_loss.item()

                total_loss_sum += total_loss.item()

                # correct num
                pred = logits.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()


        if model_type == 'kfn':
            loss_avg = loss_sum / (i+1)
            accuracy = correct / len(testloader.dataset)
            return loss_avg, accuracy

        else:
            loss_avg = loss_sum / (i+1)
            windowscls_loss_avg = windowscls_loss_sum / (i+1)
            total_loss_avg = total_loss_sum / (i+1)

            accuracy = correct / len(testloader.dataset)
            return loss_avg, windowscls_loss_avg, total_loss_avg, accuracy
