import os
import shutil
import time
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
from util_COCO import *
import torchvision.models as models
from model import model_COCO
import util

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, data_loader, optimizer=None, display=True):
        loss = self.state['loss']
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        # if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
        if display:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      '{loss:.4f}'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss=loss))
            else:
                # if data_loader=='val_unseen_loader':
                #     print("ZS!!!!!!")
                # else:
                #     print("GZS!!!!!")
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      '{loss:.4f}'.format(
                    self.state['epoch'],self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss=loss))

    def init_learning(self, model):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, train_dataset, val_dataset, wordvec_array,  unseen_ids, seen_ids, logger, optimizer=None):

        self.init_learning(model)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        # val_unseen_dataset.transform = self.state['val_transform']
        # val_unseen_dataset.target_transform = self._state('val_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        wordvec_array = torch.tensor(wordvec_array).cuda().float()
        seen_wordvec = deepcopy(wordvec_array)
        vecs_train = seen_wordvec[:, :, list(seen_ids)].squeeze().transpose(0, 1)
        print("vecs_train:",vecs_train.shape)
        # vecs_train = train_dataset[0][0][2]
        # print("vecs_train:",vecs_train.shape)
        # vecs_val_unseen = val_unseen_dataset[0][0][2]
        seen_and_unseen = seen_ids | unseen_ids
        vecs_val_seen_unseen = wordvec_array[:, :, list(seen_and_unseen)].squeeze().transpose(0, 1)
        print("vecs_val_seen_unseen:", vecs_val_seen_unseen.shape)
        # print("vecs_val_unseen:",vecs_val_unseen.shape)
        vecs_val_unseen = wordvec_array[:, :, list(unseen_ids)].squeeze().transpose(0, 1)
        print("vecs_val_unseen:", vecs_val_unseen.shape)
        # print("vecs_val_seen_unseen:", vecs_val_seen_unseen.shape)
        # train_labels = train_dataset[0][1]
        # # print("train_labels:", train_labels)
        # val_labels = val_dataset[0][1]

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=8)

        # val_unseen_loader = torch.utils.data.DataLoader(val_unseen_dataset,
        #                                          batch_size=self.state['batch_size'], shuffle=False,
        #                                          num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=8)

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            # val_seen_unseen_loader.pin_memory = True
            cudnn.benchmark = True


            # model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()



        # if self.state['evaluate']:
        #     self.validate(val_unseen_loader,val_seen_unseen_loader, vecs_val_unseen, vecs_val_seen_unseen, model, epoch)
        #     return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)

            # train for one epoch
            # self.train(train_loader,vecs_train, unseen_ids, seen_ids, model, optimizer, epoch)
            # evaluate on validation set
            self.validate(val_loader, vecs_val_unseen, vecs_val_seen_unseen, unseen_ids, seen_ids, model, epoch, logger)


    def train(self, data_loader,vecs_train, unseen_ids, seen_ids, model, optimizer, epoch):
        # args = parser.parse_args()
        # lambda_distill = 0.001
        # switch to train mode
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=0.0001,
        #                             momentum=0.9,
        #                             weight_decay=1e-4)
        model.train()
        seen_ids_tensor = torch.tensor(list(seen_ids)).cuda()
        # print("seen_ids_tensor:",seen_ids_tensor)

        self.on_start_epoch(True, model, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()


        for i, (input, target) in enumerate(data_loader):
            # measure data loading time

            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            # print("self.state['input']:",self.state['input'][1])
            target = target[:, seen_ids_tensor] #;训练过程中只用已知类
            # print("self.state['target']:", self.state['target'])
            model_vgg19 = models.vgg19(pretrained=True)
            # model_vgg19 = nn.Sequential(*list(model_vgg19.children())[:34])
            model_vgg19 = model_vgg19.features[:34]
            model_vgg19 = model_vgg19.eval()
            # model_vgg = models.vgg19(pretrained=True).features
            model_vgg19 = model_vgg19.cuda()
            # print("model:",model)
            train_inputs = model_vgg19(self.state['input'][0].cuda())
            # print("train_inputs:", train_inputs.shape)
            # print("trainbefore_self.state['target']:", self.state['target'])


            self.on_start_batch(True, model, data_loader, optimizer)

            # if self.state['use_gpu']:
            #     self.state['target'] = self.state['target'].cuda()
            # print("trainafter_self.state['target']:",self.state['target'])
            # self.on_forward(True, model, train_inputs, vecs_train, self.state['target'], optimizer)
            lambda_distill = 0.01
            # print("on_forward_labels.float():", labels)

            # model_vgg = models.vgg19(pretrained=True).features
            # # print("model:",model)
            # train_inputs = model_vgg(self.state['input'][0])
            # # print("train_inputs:", train_inputs.shape)
            # self.state['target'] = labels
            # print("self.state['target']:",self.state['target'])
            input_var = torch.autograd.Variable(train_inputs)
            target_var = torch.autograd.Variable(target)
            vecs = torch.tensor(vecs_train).cuda()
            target_var = torch.tensor(target_var).cuda()
            # print("input_var:",input_var.shape)
            # print("vecs:", vecs.shape)
            input_var = input_var.cuda()
            logits_RBF, logits_SBF, logits = model(input_var, vecs)
            # print("logits_RBF:", logits_RBF.shape)
            # print("logits_SBF:", logits_SBF.shape)
            # print("logits:", logits.shape)
            # print("target_var.float():",target_var.float().shape)
            loss_RBF = model_COCO.ranking_lossT(logits_RBF, target_var.float())  # cuda:0
            # print("loss_RBF:", loss_RBF)
            loss_SBF = model_COCO.ranking_lossT(logits_SBF, target_var.float())  # cuda:0
            # print("loss_SBF:", loss_SBF)
            # loss_SBF.backward()
            loss_distill = model_COCO.distill_loss(logits_RBF, logits_SBF)  # cuda:0
            # loss_distill = loss_distill.cuda()
            # print("loss_distill_before:", loss_distill)
            loss = loss_RBF + loss_SBF + lambda_distill * loss_distill  # cuda:0

            # print("loss:", loss)
            # if training:
            optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss) or loss.item() > 100:
                print('Unstable/High Loss:', loss)
                import pdb
                pdb.set_trace()
            optimizer.step()
            self.state['loss'] = loss
            self.state['output'] = logits

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, data_loader, optimizer)

        # self.on_end_epoch(True, model, data_loader, optimizer)
        print('Epoch: [{0}]\t''Loss {loss:.4f}\t'.format(self.state['epoch'], loss=self.state['loss']))

    def validate(self, val_loader, vecs_val_unseen, vecs_val_seen_unseen, unseen_ids, seen_ids, model, epoch, logger):
        print("validation mode")
        # switch to evaluate mode
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=0.0001,
        #                             momentum=0.9,
        #                             weight_decay=1e-4)
        model.eval()
        seen_ids_tensor = torch.tensor(list(seen_ids)).cuda()
        unseen_ids_tensor = torch.tensor(list(unseen_ids)).cuda()
        # print("unseen_ids_tensor:",unseen_ids_tensor)
        seen_unseen_ids_tensor = torch.tensor(list(seen_ids | unseen_ids)).cuda()
        # print("seen_unseen_ids_tensor:",seen_unseen_ids_tensor)
        mean_val_zs_rank_loss = 0
        mean_val_gzs_rank_loss = 0
        logits_17 = torch.empty(len(val_loader)*32-23, 17)
        target_17 = torch.empty(len(val_loader)*32-23, 17)
        logits_48_17 = torch.empty(len(val_loader)*32-23, 65)
        target_48_17 = torch.empty(len(val_loader)*32-23, 65)

        self.on_start_epoch(False, model, val_loader)

        if self.state['use_pb']:
            val_loader = tqdm(val_loader, desc='Test')

        end = time.time()

        for i, (input, target) in enumerate(val_loader): #每个i都是一个迭代
            # print("len(val_loader):",len(val_seen_unseen_loader))
            # print("i:",i)
            strt = i * 32
            endt = min(i*32 + 32, len(val_loader)*32-23)
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            target17 = target[:, unseen_ids_tensor]
            target48_17 = target[:, seen_unseen_ids_tensor]
            model_vgg19 = models.vgg19(pretrained=True)
            # model_vgg19 = nn.Sequential(*list(model_vgg19.children())[:34])
            model_vgg19 = model_vgg19.features[:34]
            model_vgg19 = model_vgg19.eval()
            # model_vgg = models.vgg19(pretrained=True).features
            model_vgg19 = model_vgg19.cuda()
            # print("model:",model)
            val_inputs = model_vgg19(self.state['input'][0].cuda())
            # print("train_val_inputs:", val_inputs.shape)

            self.on_start_batch(False, model, val_loader)

            if self.state['use_gpu']:
                target17 = target17.cuda()
                target48_17 = target48_17.cuda()

            # self.on_forward(False, model, val_inputs, vecs_val_unseen,self.state['target'])
            with torch.no_grad():
                input_var = torch.autograd.Variable(val_inputs)
                target_var17 = torch.autograd.Variable(target17)
                target_var48_17 = torch.autograd.Variable(target48_17)
                vecs_zs = torch.tensor(vecs_val_unseen).cuda()
                # print("vecs_zs:",vecs_zs.shape)
                vecs_gzs = torch.tensor(vecs_val_seen_unseen).cuda()
                target_var17 = torch.tensor(target_var17).cuda()
                target_var48_17 = torch.tensor(target_var48_17).cuda()
                # print("target_var_seen_unseen:", target_var.shape)
                # print("input_var:",input_var.shape)
                # print("vecs:", vecs.shape)
                input_var = input_var.cuda()
                # print("input_var:",input_var.shape)
                _, _, logits17 = model(input_var, vecs_zs)
                # print("logits17:",logits17)
                # print("target_var17:",target_var17)
                # print("logits17:",logits17.shape)
                _, _, logits48_17 = model(input_var, vecs_gzs)
                loss_logits_17 = model_COCO.ranking_lossT(logits17, target_var17.float())
                loss_logits_48_17 = model_COCO.ranking_lossT(logits48_17, target_var48_17.float())  # cuda:0
                logits_17[strt:endt, :] = logits17
                target_17[strt:endt, :] = target_var17
                target_48_17[strt:endt, :] = target_var48_17
                logits_48_17[strt:endt, :] = logits48_17
                mean_val_zs_rank_loss += loss_logits_17.item()
                mean_val_gzs_rank_loss += loss_logits_48_17.item()

            self.state['loss'] = loss_logits_48_17 #只输出广义的损失
            # self.state['output'] = logits
            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()

            self.on_end_batch(False, model, val_loader)

        mean_val_zs_rank_loss /= len(val_loader)
        mean_val_gzs_rank_loss /= len(val_loader)
        #compute accuracy
        ap_unseen = util.compute_AP(logits_17.cuda(), target_17.cuda())
        F1_3_unseen, P_3_unseen, R_3_unseen = util.compute_F1(logits_17.cuda(), target_17.cuda(), 'overall', k_val=3)
        mF1_3_unseen, mP_3_unseen, mR_3_unseen, map_unseen = [torch.mean(F1_3_unseen), torch.mean(P_3_unseen), torch.mean(R_3_unseen),
                                            torch.mean(ap_unseen)]
        print('k=3 AT 17',map_unseen.item(),mF1_3_unseen.item(),mP_3_unseen.item(),mR_3_unseen.item())
        ap_seen_unseen = util.compute_AP(logits_48_17.cuda(), target_48_17.cuda())
        F1_3_seen_unseen, P_3_seen_unseen, R_3_seen_unseen = util.compute_F1(logits_48_17.cuda(), target_48_17.cuda(), 'overall', k_val=3)
        mF1_3_seen_unseen, mP_3_seen_unseen, mR_3_seen_unseen, map_seen_unseen = [torch.mean(F1_3_seen_unseen), torch.mean(P_3_seen_unseen),
                                                              torch.mean(R_3_seen_unseen), torch.mean(ap_seen_unseen)]
        print('k=3 AT 48_17',map_seen_unseen.item(),mF1_3_seen_unseen.item(),mP_3_seen_unseen.item(),mR_3_seen_unseen.item())

        # values = [epoch, map_unseen, mF1_3_unseen, mP_3_unseen, mR_3_unseen, map_seen_unseen, mF1_3_seen_unseen,
        #           mP_3_seen_unseen, mR_3_seen_unseen, mean_val_zs_rank_loss, mean_val_gzs_rank_loss]
        values = [epoch, map_seen_unseen, mF1_3_seen_unseen,
                  mP_3_seen_unseen, mR_3_seen_unseen, mean_val_gzs_rank_loss]
        # values = [epoch, map_seen_unseen, mF1_3_seen_unseen, mP_3_seen_unseen, mR_3_seen_unseen]
        logger.add(values)
        logger.save()

        # remember best prec@1 and save checkpoint
        is_best = F1_3_unseen > self.state['best_score']
        self.state['best_score'] = max(F1_3_unseen, self.state['best_score'])
        self.save_checkpoint({
            'epoch': epoch + 1,
            'arch': self._state('arch'),
            'state_dict': model.state_dict() if self.state['use_gpu'] else model.state_dict(),
            'best_score': self.state['best_score'],
        }, is_best)
        print(' *** best={best:.3f}'.format(best=self.state['best_score']))


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        # self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, data_loader, optimizer)


    def on_start_batch(self, training, model, data_loader, optimizer=None, display=True):
        Engine.on_start_batch(self, training, model, data_loader, optimizer, display=False)


    def on_end_batch(self, training, model, data_loader, optimizer=None, display=True):
        Engine.on_end_batch(self, training, model, data_loader, optimizer, display=False)
        if display:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      '{loss:.4f}'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    loss=loss))
            else:
                # if data_loader = val_unseen_loader:
                #     print("ZS!!!!!!")
                # else:
                #     print("GZS!!!!!")
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      '{loss:.4f}'.format(
                    self.state['epoch'],self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    loss=loss))



