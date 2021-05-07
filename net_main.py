from torch.utils import data
from Module import *
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from torch import softmax
from Process import Processing


parser = argparse.ArgumentParser("EddyVNN")
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='init learning rate')
parser.add_argument('--train_size', type=float, default=0.6, help='size of train data set')
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--step_size', type=int, default=20, help='lr scheduler step size')
parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma value')
parser.add_argument('--model_layers', type=int, default=34, help='number of network layers')
parser.add_argument('--data_path', type=str, default='eddy_data/', help='the path of the eddy data')
parser.add_argument('--save_path', type=str, default='eddy_data/Argo/Alt purified NE/', help='the save path of the NE')
parser.add_argument('--data_type', type=str, default='Argo', help='the type of the eddies data')
parser.add_argument('--start_year', type=int, default=2002, help='the year of the first data')
parser.add_argument('--end_year', type=int, default=2003, help='final year')
args = parser.parse_args()


class Train:
    def __init__(self, path, model):
        self.path = path
        self.model = model
        self.model_running()

    def data_load(self):
        data_processing = Processing(args.data_path, args.save_path, args.start_year, args.end_year, args.data_type)
        datas, labels = data_processing.input()
        X_train, X_test, Y_train, Y_test = train_test_split(datas, labels, train_size=args.train_size)

        train_set = data.TensorDataset(torch.tensor(np.array(X_train).astype('float'), dtype=torch.float32), torch.tensor(np.array(Y_train).astype('float'), dtype=torch.int64))
        test_set = data.TensorDataset(torch.tensor(np.array(X_test).astype('float'), dtype=torch.float32), torch.tensor(np.array(Y_test).astype('float'), dtype=torch.int64))
        train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
        test_loader = data.DataLoader(test_set, batch_size=args.batch, shuffle=False)

        return train_loader, test_loader


    def model_running(self):
        train_loader, test_loader = self.data_load()
        loss_func = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)

        total_step = len(train_loader)
        Ace = []
        Loss = []
        train_ace = []

        print('start training')
        for epoch in range(args.epochs):
            cur_loss = []
            correct = 0
            for i, (datas, labels) in enumerate(train_loader):
                datas = datas.cuda()
                labels = labels.cuda()

                datas = datas.unsqueeze(1)
                datas = datas.unsqueeze(1)
                datas = datas.unsqueeze(1)

                outputs = self.model(datas)
                loss = loss_func(outputs, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                outputs = torch.max(softmax(outputs.cpu(), dim=1), dim=1)[1]
                ace = (outputs.numpy() == labels.cpu().numpy())
                correct = correct + sum(ace)

                cur_loss.append(loss.item())

            acc = (correct / (len(train_loader) * args.batch)) * 100
            train_ace.append((correct / (len(train_loader) * args.batch)) * 100)

            mean_loss = np.mean(np.array(cur_loss))
            Loss.append(mean_loss)
            print(f'Epoch: {epoch},  Loss: {mean_loss}')
            print("train accuracy：{:.3f}%".format(acc))

            if (i + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{args.epochs}, Step [{i + 1}/{total_step}], {loss.item()}')
            scheduler.step()

            model.eval()

        correct = 0
        for i, (datas, labels) in enumerate(test_loader):
            datas = datas.unsqueeze(1)
            datas = datas.unsqueeze(1)
            datas = datas.unsqueeze(1)

            datas = datas.cuda()
            outputs = self.model(datas)
            outputs = torch.max(softmax(outputs.cpu(), dim=1), dim=1)[1]
            ace = (outputs.numpy() == labels.numpy())
            correct = correct + sum(ace)

        acc = (correct / (len(test_loader) * args.batch)) * 100
        print("test accuracy：{:.3f}%".format(acc))
        Ace.append((correct / (len(test_loader) * args.batch)) * 100)


if __name__ == '__main__':
    if args.model_layers == 34:
        model = EddyVNN34().cuda()
    elif args.model_layers == 50:
        model = EddyVNN50().cuda()
    else:
        model = EddyVNN101().cuda()
    Train(args.data_path, model)

