import sys
import time
import logging
import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler 

from model.SSGRL import SSGRL
from model.intra_GCN import intra_GCN
from model.inter_GCN import inter_GCN
from loss.SSGRL_GCN_JoCoR_CST import BCELoss, KLLoss, ContrastiveLoss, getInterPseudoLabel

from utils.dataloader import get_graph_file, get_inter_graph_file, get_word_file, get_data_loader
from utils.metrics import AverageMeter, AveragePrecisionMeter, Compute_mAP_VOC2012
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint
from config import arg_parse, logger, show_args

global bestPrec1, bestPrec2, bestPrec12
bestPrec1, bestPrec2, bestPrec12 = 0, 0, 0

# vg_total_path = '/data2/wxy/vg_and_coco_test_data/data_check/vg_map_total.txt'
# vg_total_file = open(vg_total_path, 'r')
# vg_map_total = {}
# for row in vg_total_file:
#     row = row.strip('\n')
#     total = int(row[row.find(' ')+1:])
#     vg = int(row[0:row.find(' ')])
#     vg_map_total.update({vg:total})

# coco_total_path = '/data2/wxy/vg_and_coco_test_data/data_check/coco_map_total.txt'
# coco_total_file = open(coco_total_path, 'r')
# coco_map_total = {}
# for row in coco_total_file:
#     row = row.strip('\n')
#     total = int(row[row.find(' ')+1:])
#     coco = int(row[0:row.find(' ')])
#     coco_map_total.update({coco:total})

def main():
    global bestPrec1, bestPrec2, bestPrec12

    # Argument Parse
    args = arg_parse('CST')

    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = './exp/log/{}.log'.format(args.post)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Save Code File
    save_code_file(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_loader1, test_loader1 = get_data_loader(args)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")
    intraGraph1 = get_graph_file(train_loader1.dataset.intra1Labels)
    # intraGraph2 = get_graph_file(train_loader.dataset.intra2Labels)
    WordFile = get_word_file(args)
    interGraph = get_inter_graph_file(WordFile)

    model_SSGRL = SSGRL(WordFile, classNum=args.classNum)
    model_intraGCN = intra_GCN(intraGraph1, intraGraph1, classNum=args.classNum)
    model_interGCN = inter_GCN(interGraph, classNum=args.classNum)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model_SSGRL = load_pretrained_model(model_SSGRL, args)

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cpu')
        bestPrec1, args.startEpoch = checkpoint['best_mAP'], checkpoint['epoch']
        bestPrec2 = bestPrec1
        bestPrec12 = bestPrec1
        model_SSGRL.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}, mAP: {1}".format(args.startEpoch, bestPrec1))

    model_SSGRL.cuda()
    model_intraGCN.cuda()
    model_interGCN.cuda()
    logger.info("==> Done!\n")

    criterion = {'BCELoss': BCELoss(reduce=True, size_average=True).cuda(),
                 'KLLoss': KLLoss(reduce=True, size_average=True, co_lambda=args.co_lambda).cuda(),
                 'PseudoBCELoss': BCELoss(margin=args.pseudoBCEMargin, reduce=True, size_average=True).cuda(),
                 'PseudoDistanceLoss': ContrastiveLoss(args.batchSize, reduce=True, size_average=True).cuda(),
                }

    for p in model_SSGRL.backbone.parameters():
        p.requires_grad = False
    for p in model_SSGRL.backbone.layer4.parameters():
        p.requires_grad = True
    optimizer1 = torch.optim.Adam(filter(lambda p : p.requires_grad, model_SSGRL.parameters()), lr=args.lr, weight_decay=args.weightDecay)
    optimizer2 = torch.optim.Adam(filter(lambda p : p.requires_grad, model_intraGCN.parameters()), lr=args.lr, weight_decay=args.weightDecay)
    optimizer3 = torch.optim.Adam(filter(lambda p : p.requires_grad, model_interGCN.parameters()), lr=args.lr, weight_decay=args.weightDecay)

    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.stepEpoch, gamma=0.1)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.stepEpoch, gamma=0.1)
    scheduler3 = lr_scheduler.StepLR(optimizer3, step_size=args.stepEpoch, gamma=0.1)

    rate_schedule = np.ones(args.epochs) * args.forget_rate
    rate_schedule[:args.num_gradual] = np.linspace(0, args.forget_rate ** args.exponent, args.num_gradual)

    if args.evaluate:
        Validate(test_loader1, model_SSGRL, model_intraGCN, model_interGCN, criterion, 0, args)
        return

    logger.info('Total: {:.3f} GB'.format(torch.cuda.get_device_properties(0).total_memory/1024.0**3))

    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('./exp/summary/', args.post))

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):

        intraA1 = model_intraGCN.intraAdj1.detach().clone().cpu()
        intraA2 = model_intraGCN.intraAdj2.detach().clone().cpu()
        interA = model_interGCN.interAdj.detach().clone().cpu()
        np.save('./txt/{}/intraA1_{}.npy'.format(args.post, epoch), intraA1)
        np.save('./txt/{}/intraA2_{}.npy'.format(args.post, epoch), intraA2)
        np.save('./txt/{}/interA_{}.npy'.format(args.post, epoch), interA)

        Train(train_loader1, model_SSGRL, model_intraGCN, model_interGCN, criterion, optimizer1, optimizer2, optimizer3, rate_schedule, writer, epoch, args)
        val1_mAP1, val1_mAP2, val1_mAP12 = Validate(test_loader1, model_SSGRL, model_intraGCN, model_interGCN, criterion, epoch, args)
        # val2_mAP1, val2_mAP2, val2_mAP12 = Validate(test_loader2, model_SSGRL, model_intraGCN, model_interGCN, criterion, epoch, args)
        # mAP1 = ( val1_mAP1 + val2_mAP1 ) / 2
        # mAP2 = ( val1_mAP2 + val2_mAP2 ) / 2
        # mAP12 = ( val1_mAP12 + val2_mAP12 ) / 2
        mAP1 = val1_mAP1
        mAP2 = val1_mAP2
        mAP12 = val1_mAP12
        
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        writer.add_scalar('mAP1', mAP1, epoch)
        writer.add_scalar('mAP2', mAP2, epoch)
        writer.add_scalar('mAP12', mAP12, epoch)
        torch.cuda.empty_cache()

        isBest1, bestPrec1 = mAP1 > bestPrec1, max(mAP1, bestPrec1)
        save_checkpoint('model_SSGRL', {'epoch':epoch, 'state_dict':model_SSGRL.state_dict(), 'best_mAP1':mAP1}, isBest1)
        isBest2, bestPrec2 = mAP2 > bestPrec2, max(mAP2, bestPrec2)
        save_checkpoint('model_intraGCN', {'epoch':epoch, 'state_dict':model_intraGCN.state_dict(), 'best_mAP2':mAP2}, isBest2)
        isBest12, bestPrec12 = mAP12 > bestPrec12, max(mAP12, bestPrec12)
        save_checkpoint('model_interGCN', {'epoch':epoch, 'state_dict':model_interGCN.state_dict(), 'best_mAP':mAP12}, isBest12)


        if isBest1:
            logger.info('[Best] [Epoch {0}]: Best mAP1 is {1:.3f}'.format(epoch, bestPrec1))
        if isBest2:
            logger.info('[Best] [Epoch {0}]: Best mAP2 is {1:.3f}'.format(epoch, bestPrec2))
        if isBest12:
            logger.info('[Best] [Epoch {0}]: Best mAP12 is {1:.3f}'.format(epoch, bestPrec12))

    writer.close()

def Train(train_loader, model_SSGRL, model_intraGCN, model_interGCN, criterion, optimizer1, optimizer2, optimizer3, rate_schedule, writer, epoch, args):

    model_SSGRL.train()
    model_intraGCN.train()
    model_interGCN.train()
    model_SSGRL.backbone.eval()
    model_SSGRL.backbone.layer4.train()

    loss_ssgrl, loss_intra, loss_inter, loss1, loss2, loss3, loss4, loss5, loss6, loss7  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(train_loader):
        input, target = input.cuda(), target.float().cuda()

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        semanticFeature = model_SSGRL(input)
        output_intra = model_intraGCN(input, semanticFeature)
        output_inter = model_interGCN(input, semanticFeature) 

        model_SSGRL.updateFeature(semanticFeature, target, args.pseudoExampleNumber)
        pseudoTarget = getInterPseudoLabel(semanticFeature, target,
                                  model_SSGRL.posFeature,
                                  margin=args.pseudoBCEMargin) if epoch >= args.generateLabelEpoch else target

        # Compute and log loss
        loss1_ = criterion['BCELoss'](output_intra, target)
        loss5_ = args.pseudoBCEWeight * criterion['PseudoBCELoss'](output_intra, pseudoTarget) if epoch >= args.generateLabelEpoch else \
                 0 * criterion['PseudoBCELoss'](output_intra, pseudoTarget)
        loss2_ = criterion['BCELoss'](output_inter, target)
        loss6_ = args.pseudoBCEWeight * criterion['PseudoBCELoss'](output_inter, pseudoTarget) if epoch >= args.generateLabelEpoch else \
                 0 * criterion['PseudoBCELoss'](output_inter, pseudoTarget)
        loss7_ = args.pseudoDistanceWeight * criterion['PseudoDistanceLoss'](semanticFeature, target) if epoch >= 1 else \
                 args.pseudoDistanceWeight * criterion['PseudoDistanceLoss'](semanticFeature, target) * batchIndex / float(len(train_loader))
        loss3_ = criterion['KLLoss'](output_intra, output_inter.detach(), target, pseudoTarget, args.pseudoBCEWeight, epoch, args.generateLabelEpoch, rate_schedule[epoch]) 
        # if epoch >= args.generateLabelEpoch else \
        #          0 * criterion['KLLoss'](output_intra, output_inter.detach(), target, pseudoTarget, args.pseudoBCEWeight, epoch, args.generateLabelEpoch, rate_schedule[epoch])
        loss4_ = criterion['KLLoss'](output_inter, output_intra.detach(), target, pseudoTarget, args.pseudoBCEWeight, epoch, args.generateLabelEpoch, rate_schedule[epoch])
        #  if epoch >= args.generateLabelEpoch else \
        #          0 * criterion['KLLoss'](output_inter, output_intra.detach(), target, pseudoTarget, args.pseudoBCEWeight, epoch, args.generateLabelEpoch, rate_schedule[epoch])
        loss_ssgrl_ = 1.0 * (loss1_ + loss5_) + 1.0 * (loss2_ + loss6_) + loss7_
        loss_intra_ = loss3_
        loss_inter_ = loss4_

        loss1.update(loss1_.item(), input.size(0))
        loss2.update(loss2_.item(), input.size(0))
        loss3.update(loss3_.item(), input.size(0))
        loss4.update(loss4_.item(), input.size(0))
        loss5.update(loss5_.item(), input.size(0))
        loss6.update(loss6_.item(), input.size(0))
        loss7.update(loss7_.item(), input.size(0))

        loss_ssgrl.update(loss_ssgrl_.item(), input.size(0))
        loss_intra.update(loss_intra_.item(), input.size(0))
        loss_inter.update(loss_inter_.item(), input.size(0))

        # Backward
        optimizer2.zero_grad()         
        optimizer3.zero_grad()                          
        loss3_.backward(retain_graph=True)          
        loss4_.backward(retain_graph=True) 

        optimizer1.zero_grad()                      
        loss_ssgrl_.backward(retain_graph=True)       

        optimizer1.step()       
        optimizer2.step()
        optimizer3.step()                    

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        if batchIndex % args.printFreq == 0:
            logger.info('[Train] [Epoch {0}]: [{1:04d}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f}\n'
                        '\t\t\t\t\tLearn Rate1 {lr1:.6f} Learn Rate2 {lr2:.6f} Learn Rate3 {lr3:.6f}\n'
                        '\t\t\t\t\tBCE Loss1 {loss1.val:.4f} ({loss1.avg:.4f}) BCE Loss2 {loss2.val:.4f} ({loss2.avg:.4f}) KL Loss1 {loss3.val:.4f} ({loss3.avg:.4f}) KL Loss2 {loss7.val:.4f} ({loss7.avg:.4f})\n'
                        '\t\t\t\t\tPseudoIntraBCE Loss1 {loss8.val:.4f} ({loss8.avg:.4f}) PseudoInterBCE Loss2 {loss9.val:.4f} ({loss9.avg:.4f}) PseudoDistance Loss1 {loss10.val:.4f} ({loss10.avg:.4f})\n'
                        '\t\t\t\t\tSSGRL Loss {loss4.val:.4f} ({loss4.avg:.4f}) Intra Loss {loss5.val:.4f} ({loss5.avg:.4f}) Inter Loss {loss6.val:.4f} ({loss6.avg:.4f})'.format(
                        epoch, batchIndex, len(train_loader), batch_time=batch_time, data_time=data_time,
                        lr1=optimizer1.param_groups[0]['lr'], lr2=optimizer2.param_groups[0]['lr'], lr3=optimizer3.param_groups[0]['lr'],
                        loss1=loss1, loss2=loss2, loss3=loss3, loss4=loss_ssgrl, loss5=loss_intra, loss6=loss_inter, loss7=loss4, loss8=loss5, loss9=loss6, loss10=loss7))
            sys.stdout.flush()

    writer.add_scalar('Loss1', loss1.avg, epoch)
    writer.add_scalar('Loss2', loss2.avg, epoch)
    writer.add_scalar('Loss3', loss3.avg, epoch)
    writer.add_scalar('Loss4', loss4.avg, epoch)
    writer.add_scalar('Loss5', loss5.avg, epoch)
    writer.add_scalar('Loss6', loss6.avg, epoch)
    writer.add_scalar('Loss7', loss7.avg, epoch)
    writer.add_scalar('Loss_SSGRL', loss_ssgrl.avg, epoch)
    writer.add_scalar('Loss_Intra', loss_intra.avg, epoch)
    writer.add_scalar('Loss_Inter', loss_inter.avg, epoch)


def Validate(val_loader, model_SSGRL, model_intraGCN, model_interGCN, criterion, epoch, args):

    model_SSGRL.eval()
    model_intraGCN.eval()
    model_interGCN.eval()

    apMeter1, apMeter2, apMeter12 = AveragePrecisionMeter(), AveragePrecisionMeter(), AveragePrecisionMeter()
    pred1, pred2, pred12, loss1, loss2, loss12, batch_time, data_time = [], [], [], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()

    for batchIndex, (sampleIndex, input, target, groundTruth) in enumerate(val_loader):
        input, target = input.cuda(), target.float().cuda()
        
        # Log time of loading data
        data_time.update(time.time()-end)

        # Forward
        with torch.no_grad():
            # output, intraCoOccurrence, feature = model(input)
            semanticFeature = model_SSGRL(input)
            output_intra = model_intraGCN(input, semanticFeature)
            output_inter = model_interGCN(input, semanticFeature)
            output_avg = output_intra * 0.5 + output_inter * 0.5

        # Compute loss and prediction
        loss1_ = criterion['BCELoss'](output_intra, target)
        loss2_ = criterion['BCELoss'](output_inter, target)
        loss12_ = criterion['BCELoss'](output_avg, target)
        loss1.update(loss1_.item(), input.size(0))
        loss2.update(loss2_.item(), input.size(0))
        loss12.update(loss12_.item(), input.size(0))

        # Change target to [0, 1]
        target[target < 0] = 0

        apMeter1.add(output_intra, target)
        apMeter2.add(output_inter, target)
        apMeter12.add(output_avg, target)
        pred1.append(torch.cat((output_intra, (target>0).float()), 1))
        pred2.append(torch.cat((output_inter, (target>0).float()), 1))
        pred12.append(torch.cat((output_avg, (target>0).float()), 1))

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info information of current batch        
        if batchIndex % args.printFreq == 0:
            logger.info('[Test] [Epoch {0}]: [{1:04d}/{2}] '
                        'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'Loss1 {loss1.val:.4f} ({loss1.avg:.4f}) Loss2 {loss2.val:.4f} ({loss2.avg:.4f}) Loss12 {loss12.val:.4f} ({loss12.avg:.4f})'.format(
                epoch, batchIndex, len(val_loader),
                batch_time=batch_time, data_time=data_time,
                loss1=loss1, loss2=loss2, loss12=loss12))
            sys.stdout.flush()

    pred1 = torch.cat(pred1, 0).cpu().clone().numpy()
    pred2 = torch.cat(pred2, 0).cpu().clone().numpy()
    pred12 = torch.cat(pred12, 0).cpu().clone().numpy()
    mAP1, AP1 = Compute_mAP_VOC2012(pred1, args.classNum)
    mAP2, AP2 = Compute_mAP_VOC2012(pred2, args.classNum)
    mAP12, AP12 = Compute_mAP_VOC2012(pred12, args.classNum)
    averageAP1 = apMeter1.value().mean()
    averageAP2 = apMeter2.value().mean()
    averageAP12 = apMeter12.value().mean()

    OP, OR, OF1, CP, CR, CF1 = apMeter1.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter1.overall_topk(3)
    logger.info('[Test] mAP1: {mAP:.3f}, averageAP1: {averageAP:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
                mAP=mAP1, averageAP=averageAP1,
                OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))
    for i, ap in enumerate(AP1):
        logger.info('{I:.0f} AP1 = {AP1:.3f}'.format(I = i, AP1 = ap))

    OP, OR, OF1, CP, CR, CF1 = apMeter2.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter2.overall_topk(3)
    logger.info('[Test] mAP2: {mAP:.3f}, averageAP2: {averageAP:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
                mAP=mAP2, averageAP=averageAP2,
                OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))
    for i, ap in enumerate(AP2):
        logger.info('{I:.0f} AP2 = {AP2:.3f}'.format(I = i, AP2 = ap))


    OP, OR, OF1, CP, CR, CF1 = apMeter12.overall()
    OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = apMeter12.overall_topk(3)
    logger.info('[Test] mAP12: {mAP:.3f}, averageAP12: {averageAP:.3f}\n'
                '\t\t\t\t\t(Compute with all label) OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}, CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}\n'
                '\t\t\t\t\t(Compute with top-3 label) OP: {OP_K:.3f}, OR: {OR_K:.3f}, OF1: {OF1_K:.3f}, CP: {CP_K:.3f}, CR: {CR_K:.3f}, CF1: {CF1_K:.3f}'.format(
                mAP=mAP12, averageAP=averageAP12,
                OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, OP_K=OP_K, OR_K=OR_K, OF1_K=OF1_K, CP_K=CP_K, CR_K=CR_K, CF1_K=CF1_K))
    for i, ap in enumerate(AP12):
        logger.info('{I:.0f} AP12 = {AP12:.3f}'.format(I = i, AP12 = ap))


    return mAP1, mAP2, mAP12


if __name__=="__main__":
    main()
