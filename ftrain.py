from datasets import *
from architectures import *
from Learning.learning import *
from Learning.losses import *
from Utils.parser_ import get_args
from fastai2.vision.all import *
from Learning.learning import test
from utils_ import measure_time

@dataclass
class Do_everything(Callback):
    learn: Learner

    def begin_epoch(self):  # on_epoch_begin
        L = self.learn
        if hasattr(self, 'skip_prepare') and self.skip_prepare:
            return
        # L.model.drop_path_prob = args.drop_path_prob * L.epoch / args.epochs
        L.model.drop_path_prob = 0.2 * L.epoch / args.epochs
        L.dls.loaders[0].prepare_epoch()

    def begin_batch(self):  # before the output and loss are computed. # on_batch_begin
        L = self.learn
        self.info = L.yb

    def after_pred(self):  # after forward pass but before loss has been computed # on_loss_begin
        pass

    def after_loss(self): # after the forward pass and the loss has been computed, but before backprop # on_backward_begin
        L = self.learn
        L.edge = L.loss[1]
        L.loss = L.loss[0]

    def after_train(self):  # on_epoch_end
        L = self.learn
        if (args.all_info or L.epoch==args.epochs-1):
            model_dir = pjoin(args.model_dir, save_name)
            os.makedirs(model_dir, exist_ok=True)
            save_path = pjoin(model_dir, 'checkpoint_{}.pt'.format(L.epoch))
            printc.green('saving to: {} ...'.format(save_path))
            torch.save({'epoch': L.epoch + 1, 'state_dict': L.model.state_dict(), 'optimizer': L.opt_func, 'model_arch': L.model.name, 'save_name':save_name}, save_path)
            dst = pjoin(model_dir, '{}.pt'.format(save_name))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.relpath(save_path, getdir(dst)), dst)

    def begin_validate(self):  # on_epoch_end
        L = self.learn
        # if not args.AMOS_RGB and not args.depths:
        if not args.notest:
            for test_loader in L.test_loaders:
                test(test_loader, L.model, test_loader.dataset.name)
        raise CancelValidException()

def our_loss(name:str, embeddings, info):
    loss, edge, pos, neg = tripletMargin_original(anchor=embeddings[0::2], positive=embeddings[1::2],
                                                  margin_pos=args.marginpos,
                                                  batch_reduce=args.batch_reduce,
                                                  loss_type=name,
                                                  detach_neg=args.detach,
                                                  get_edge=True,
                                                  block_sizes=info['block_sizes'] if 'block_sizes' in info.keys() else None,
                                                  dup_protection = not args.closeok)
    return loss.mean(), edge
    # return torch.sum(loss), edge

# def our_loss_(name:str, miner, embeddings, info):
#     labels = info['labels'].long().cuda()
#     hard_pairs = miner(embeddings, labels)
#     loss, edge = our_tripletMargin_(embeddings=embeddings, labels=labels, indices_tuple=hard_pairs,
#                                        margin_pos=args.marginpos,
#                                        detach_neg=args.detach,
#                                        get_edge=True,
#                                        block_sizes=info['block_sizes'] if 'block_sizes' in info.keys() else None,)
#     return loss.mean(), edge

def our_loss_generalized(name:str, embeddings, info):
    labels = info['labels'].long().cuda()
    # hard_pairs = miner(embeddings, labels)
    loss, edge = tripletMargin_generalized(embeddings=embeddings, labels=labels,
                                           margin_pos=args.marginpos,
                                           # detach_neg=args.detach,
                                           # get_edge=True,
                                           # block_sizes=info['block_sizes'] if 'block_sizes' in info.keys() else None
                                           )
    return loss.mean(), edge
    # return torch.sum(loss), edge

def softMarginLoss(loss_fc, embeddings, info):
    loss = loss_fc(embeddings)
    return loss, None

def their_loss(loss_fc, miner, embeddings, info):
    labels = info['labels'].long().cuda()
    hard_pairs = miner(embeddings, labels)
    loss = loss_fc(embeddings, labels, hard_pairs)
    return loss, None

def load_model(model_arch):
    model = get_model_by_name(model_arch).cuda()
    if args.resume != '':
        printc.green('Loading', args.resume, '...')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def get_loss_function(name, num_classes, embedding_size=None, miner=None):
    if name in ['face']:
        loss = CosFaceLoss(margin=args.face_margin, num_classes=num_classes, embedding_size=embedding_size, scale=args.face_scale)
        loss.cuda()
    elif name in ['theirTripletMargin']:
        if miner in ['BatchHardMiner']:
            loss = partial(their_loss, TripletMarginLoss(margin=1.0), miners.BatchHardMiner())
        elif miner in ['DistanceWeightedMiner']:
            loss = partial(their_loss, TripletMarginLoss(margin=1.0), miners.DistanceWeightedMiner(0.5, 1.0))
        elif miner in ['PairMarginMiner']:
            loss = partial(their_loss, TripletMarginLoss(margin=1.0), miners.PairMarginMiner(0.5, 0.5, False))
        elif miner in ['TripletMarginMiner']:
            loss = partial(their_loss, TripletMarginLoss(margin=1.0), miners.TripletMarginMiner(1.0))
    elif name in 'softMargin':
        print('using softMargin')
        loss = partial(softMarginLoss, DynamicSoftMarginLoss())
    elif name[-2:]=='++':
        printc.red('using new FASTER implem')
        loss = partial(our_loss_generalized, name[:-2]) # new implementation
    # elif name[-1]=='+':
    #     printc.red('using new implem')
    #     loss = partial(our_loss_, name[:-1], miners.BatchHardMiner()) # new implementation
    else:
        assert args.Npos == 2
        loss = partial(our_loss, name)
    return loss

def main(train_loader, test_loaders, model_arch):
    model = load_model(model_arch)
    data = DataLoaders(train_loader, test_loaders[0])
    loss = get_loss_function(args.loss, train_loader.total_num_labels, miner=args.miner, embedding_size=model.osize if hasattr(model,'osize') else None)
    L = Learner(data, model, loss_func=loss, metrics=[], cbs=[Do_everything]) # opt_func=SGD,
    L.test_loaders = test_loaders
    L.fit_one_cycle(args.epochs, args.lr)


if __name__ == '__main__':
    with measure_time():
        args, data_name, save_name = get_args()
        KoT.set_kornia_tr(args.K) # this turns on Kornia transform if it was set in parser
        become_deterministic(args.seed)
        print('data_name:', data_name, '\nsave_name:', save_name, '\n')
        main(train_loader=get_train_dataset(args, data_name).init(args.model_dir, save_name, args=args),
             test_loaders=get_test_loaders(['liberty'], args.batch_size),
             model_arch=args.model_arch)
        printc.green('--------------- Training finished ---------------')
        print('model_name:', save_name)