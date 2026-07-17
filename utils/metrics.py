import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score, jaccard_score

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def postprocess_small_regions(masks, min_area: int=100):
    from models.sam.utils.amg import remove_small_regions

    new_masks = []
    scores = []

    for mask in masks:
        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(mask)
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    masks = np.array(new_masks).astype('int32')
    return masks

def eval_seg(pred,true_mask_p,threshold, min_area=100):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    pred = F.sigmoid(pred)
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        processed_pred = (pred > 0.5).to(pred.dtype)
        return (iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)), processed_pred
    elif c > 2: # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred_channel = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
                '''iou for numpy'''
                ious[i] += iou(pred_channel,mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
        processed_pred = (pred > 0.5).to(pred.dtype)
        return tuple(np.array(ious + dices) / len(threshold)), processed_pred
    else:
        eiou, edice = 0,0

        # Additional metrics for binary segmentation
        eaccuracy, esensitivity, especificity, eauc, emcc, ef1, ejaccard = (0,) * 7 

        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            # Post-processing on small regions
            if min_area > 0:
                # print(disc_pred.shape, disc_pred.max())
                disc_pred = postprocess_small_regions(disc_pred, min_area=min_area)
                # print(disc_pred.shape, disc_pred.max())

            disc_mask = gt_vmask_p [:,0,:,:].cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            # Flatten arrays for scikit-learn calculations across the whole batch
            y_pred_flat = disc_pred.ravel()
            y_test_flat = disc_mask.ravel()

            '''dice using sklearn f1_score'''
            edice += f1_score(y_test_flat, y_pred_flat, zero_division=0)
            
            # Compute confusion matrix and metrics using scikit-learn built-in methods
            tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_flat, labels=[0, 1]).ravel()
            eaccuracy += accuracy_score(y_test_flat, y_pred_flat)
            esensitivity += recall_score(y_test_flat, y_pred_flat, zero_division=0)
            especificity += tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            if len(np.unique(y_test_flat)) > 1:
                y_pred_image = pred[:,0,:,:].cpu().numpy().ravel()
                eauc += roc_auc_score(y_test_flat, y_pred_image)
                # eauc += roc_auc_score(y_test_flat, y_pred_flat)
            else:
                eauc += accuracy_score(y_test_flat, y_pred_flat)
                
            emcc += matthews_corrcoef(y_test_flat, y_pred_flat)
            ef1 += f1_score(y_test_flat, y_pred_flat, zero_division=0)
            ejaccard += jaccard_score(y_test_flat, y_pred_flat, zero_division=0)

        # Get post-processed prediction mask at threshold 0.5
        vpred_vis = (pred > 0.5).float()
        processed_pred = vpred_vis
        if min_area > 0:
            disc_pred_vis = vpred_vis.cpu()[:,0,:,:].numpy().astype('int32')
            disc_pred_vis = postprocess_small_regions(disc_pred_vis, min_area=min_area)
            processed_pred = torch.as_tensor(disc_pred_vis, dtype=pred.dtype, device=pred.device).unsqueeze(1)

        return (
            eiou / len(threshold), 
            edice / len(threshold),
            eaccuracy / len(threshold), 
            esensitivity / len(threshold), 
            especificity / len(threshold), 
            eauc / len(threshold), 
            emcc / len(threshold), 
            ef1 / len(threshold), 
            ejaccard / len(threshold),
        ), processed_pred

