import torch
import numpy as np
import torchnet as tnt
from torch.autograd import Variable
from model import TripletNet, TripletNet2
from dataloader import MSD_Dataset 
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import argparse


parser= argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, help='WHICH GPU')
args = parser.parse_args()
print (args)
args.cuda = torch.cuda.is_available()
print (args.cuda)


# parameters
batch_size = 1
learning_rate = 0.001
num_epochs = 10
loss_margin = 1.0
dropout_rate = 0.5

# logger
port = 8097
train_loss_logger = VisdomPlotLogger('line', port=port, opts={'title':'Train Loss'})
valid_loss_logger = VisdomPlotLogger('line', port=port, opts={'title':'Valid Loss'})
test_loss_logger = VisdomPlotLogger('line', port=port, opts={'title':'Test Loss'})
train_acc_logger = VisdomPlotLogger('line', port=port, opts={'title':'Train Accuracy'})
valid_acc_logger = VisdomPlotLogger('line', port=port, opts={'title':'Valid Accuracy'})
test_acc_logger = VisdomPlotLogger('line', port=port, opts={'title':'Test Accuracy'})

''' Calculate if the model output has correct distance measurement. 1 if correct, 0 elsewise'''
def get_score(out_anchor, out_pos, out_neg, margin):
    pos_dist = F.pairwise_distance(out_anchor, out_pos)
    neg_dist = F.pairwise_distance(out_anchor, out_neg)
    if ((pos_dist.data[0] + margin) < neg_dist.data[0]).all() :
        return 1
    else :
        return 0

def train(model, train_loader, val_loader, criterion, learning_rate, num_epochs, loss_margin, save_dir, args):
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, verbose=True)

    # Train network
    model.train()
    for epoch in range(num_epochs):
        score = 0
        for i, triplet in enumerate(train_loader):
            anchor = Variable(triplet['pos1'])
            pos = Variable(triplet['pos2'])
            neg = Variable(triplet['neg'])

            if args.cuda:
                anchor = anchor.cuda(args.device_num)
                pos = pos.cuda(args.device_num)
                neg = neg.cuda(args.device_num)

            optimizer.zero_grad()
            out_anchor, out_pos, out_neg = model(anchor, pos, neg)
            loss = criterion(out_anchor, out_pos, out_neg)
            loss.backward()
            optimizer.step()

            score += get_score(out_anchor, out_pos, out_neg, loss_margin)
            curr_score = score / (i+1)
                
            if (i+1)%10 == 0:
                print ("Epoch %d/%d [%d/%d] Loss : %.4f, Accuracy : %.4f"%(epoch+1,num_epochs, i+1, len(train_loader), loss.data[0], curr_score))
                # log the loss and accuracy with visdom
                train_loss_logger.log(epoch*len(train_loader) + (i+1), loss.data[0])
                train_acc_logger.log(epoch*len(train_loader) + (i+1), curr_score)
        
        print ("Evaluating..")
        eval(model, val_loader, criterion, loss_margin, epoch, args)
        lr_scheduler.step(curr_score) 
    
    # Save trained model
    torch.save(model.state_dict(), 'TripletNet.pt')

def eval(model, val_loader, criterion, loss_margin, epoch, args) :
    model.eval()
    total_loss = 0
    score = 0
    for i, triplet in enumerate(val_loader):
        anchor = Variable(triplet['pos1'], volatile=True)
        pos = Variable(triplet['pos2'], volatile=True)
        neg = Variable(triplet['neg'], volatile=True)

        if args.cuda:
            anchor = anchor.cuda(args.device_num)
            pos = pos.cuda(args.device_num)
            neg = neg.cuda(args.device_num)
        
        out_anchor, out_pos, out_neg = model(anchor, pos, neg)
        loss = criterion(out_anchor, out_pos, out_neg)
        total_loss += loss.data[0]
        score += get_score(out_anchor, out_pos, out_neg, loss_margin)

    total_loss = total_loss/len(val_loader)
    accuracy = score / len(val_loader)
    valid_loss_logger.log(epoch, total_loss)
    valid_acc_logger.log(epoch, accuracy)
    
    print ("Eval Loss: %.4f, Accuray : %.4f" %(total_loss, accuracy))


def main () :
    # load data
    base_dir = '/media/bach4/kylee/triplenet-data/'
    save_dir = base_dir + 'saved_model/'
    
    print ("Start loading data..")
    train_data = MSD_Dataset(base_dir + 'mel_dir/', base_dir + 'train.txt')
    valid_data = MSD_Dataset(base_dir + 'mel_dir/', base_dir + 'valid.txt')
    test_data = MSD_Dataset(base_dir + 'mel_dir/', base_dir + 'test.txt')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    print ("Data loaded : Train %d, valid %d, test %d"%(len(train_data), len(valid_data), len(test_data)))

    # load model
    model = TripletNet2(dropout_rate)
    
    if args.cuda:
        model.cuda(args.device_num)
   
    criterion = torch.nn.TripletMarginLoss(margin=loss_margin)
    print ("Start training..")
    train(model, train_loader, valid_loader, criterion, learning_rate, num_epochs, loss_margin, save_dir, args)
    
    print ("Training done. Test on test data..")
    eval(model, test_loader, criterion,loss_margin, 0, args) 


if __name__=='__main__':
    main()
    
