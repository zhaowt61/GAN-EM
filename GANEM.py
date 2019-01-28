import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
import json
class generator(nn.Module):

    def __init__(self,class_num, dataset = 'mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62 + class_num
            self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):

        x=input
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):

    def __init__(self, class_num,dataset = 'mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1 + 10
            self.output_dim = 1
            self.class_num=class_num
        self.conv = nn.Sequential(

            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(

            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.class_num+1),


        )

        utils.initialize_weights(self)

    def forward(self, input):

        x=input
        x = self.conv(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc(x)

        return x

class E_net(nn.Module):

    def __init__(self, class_num, dataset = 'mnist'):
        super(E_net, self).__init__()
        if dataset == 'mnist' or 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1 + 10
            self.output_dim = 1

        self.conv = nn.Sequential(


            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),


        )
        self.fc = nn.Sequential(

            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, class_num),

        )

        utils.initialize_weights(self)

    def forward(self, input):

        x=input
        x = self.conv(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc(x)

        return x


class GANEM(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.epoch = args.epoch
        self.visual_num = 100 # visualization number
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type

        self.data_preprocess()
        # load mnist

        # networks init
        self.E = E_net(class_num=self.class_num, dataset=self.dataset)
        self.G = generator(class_num=self.class_num,dataset=self.dataset)
        self.D = discriminator(class_num=self.class_num,dataset=self.dataset)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        #print('---------- Networks architecture -------------')
        #utils.print_network(self.G)
        #utils.print_network(self.D)
        #print('-----------------------------------------------')


        '''
        plt.imshow(torch.squeeze(x[13001].cpu()).numpy(), cmap='gray')
        plt.title('caonima')
        plt.show()
        '''
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.D.train()
        self.E.train()

        print('training start!!')
        start_time = time.time()
        first_time=True
        self.accuracy_hist=[]

        for epoch in range(self.epoch):
            self.G.train()  # check here!!
            epoch_start_time = time.time()
            decay=0.98**epoch
            self.E_optimizer = optim.Adam(self.E.parameters(), lr=decay * 0.3 * self.args.lrD,
                                          betas=(self.args.beta1, self.args.beta2))
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=decay * 3 * self.args.lrG, betas=(self.args.beta1, self.args.beta2))
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=decay * self.args.lrD, betas=(self.args.beta1, self.args.beta2))
            for M_epoch in range(5):
                for iter, (batch_x, batch_y) in enumerate(self.train_loader):

                    x_=batch_x
                    z_=torch.rand((self.batch_size, self.z_dim))
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                    G_batch_size = batch_x.size()[0]
                    if G_batch_size < self.batch_size:
                        break
                    # x_  (batch, 1L, 28L, 28L)
                    # z_  (batch, 62L)

                    # update D network:

                    image_real = Variable(batch_x.cuda())
                    self.E.eval()
                    y_real = self.E(image_real)
                    y_real = nn.Softmax()(y_real)
                    y_real = (y_real).data.cpu().numpy()  #


                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    if first_time:
                        y_real = (1 / float(self.class_num)) * np.ones((G_batch_size, self.class_num)) # first_time

                    y_real = np.concatenate((y_real, 2*np.ones((np.shape(y_real)[0], 1))), axis=1)

                    ones=np.ones((np.shape(y_real)[0],np.shape(y_real)[1]))
                    ones[:,-1]=0
                    ones=torch.FloatTensor(ones)
                    ones=Variable(ones).cuda()
                    y_real=torch.FloatTensor(y_real).cuda()

                    D_real_loss = torch.nn.BCEWithLogitsLoss(weight=y_real)(D_real,ones)

                    G_input, conditional_label = self.gen_cond_label(self.batch_size)
                    G_ = self.G(G_input, 0)
                    D_fake = self.D(G_)
                    y_fake_1 = np.tile(np.zeros((self.class_num)), (self.batch_size, 1))
                    y_fake_2 = np.tile(np.ones((1)), (self.batch_size, 1))
                    y_fake = np.concatenate((y_fake_1,y_fake_2),axis=1)
                    y_fake = Variable(torch.FloatTensor(y_fake).cuda())
                    D_fake_loss=torch.nn.BCEWithLogitsLoss()(D_fake,y_fake)

                    D_loss = D_real_loss + D_fake_loss

                    self.train_hist['D_loss'].append(D_loss.data[0])
                    D_loss.backward()
                    self.D_optimizer.step()


                    # update G network:

                    self.G_optimizer.zero_grad()
                    G_input, conditional_label = self.gen_cond_label(self.batch_size)
                    G_ = self.G(G_input, 0)
                    D_fake = self.D(G_)

                    G_y_real=np.concatenate((conditional_label.numpy(),np.tile([0],(self.batch_size,1))),axis=1)
                    G_y_real=Variable(torch.FloatTensor(G_y_real)).cuda()
                    G_loss=torch.nn.BCEWithLogitsLoss()(D_fake,G_y_real)


                    self.train_hist['G_loss'].append(G_loss.data[0])
                    G_loss.backward()
                    self.G_optimizer.step()

                    if ((iter + 1) % 100) == 0:
                        print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.data[0], G_loss.data[0]))



            self.E_training(200)
            first_time = False
            self.visualize_results((epoch+1))
            self.compute_accuracy()
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.save()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=False):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """

            G_input, conditional_label = self.gen_cond_label(100)
            samples = self.G(G_input,0)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        #torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        #torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_E.pkl'))

        torch.save(self.G, os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D, os.path.join(save_dir, self.model_name + '_D.pkl'))
        torch.save(self.E, os.path.join(save_dir, self.model_name + '_E.pkl'))


        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_E.pkl')))

    def gen_cond_label(self,points_num):
        num = 0
        for i in range(self.class_num):
            class_i = np.zeros((self.class_num))
            class_i[i] = 1

            class_i_num = round(points_num * (1 / float(self.class_num)))

            if i == self.class_num - 1:
                class_i_num = points_num - num
            if i == 0:
                conditional_label = np.tile(class_i, (int(class_i_num), 1))
            else:
                conditional_label = np.concatenate((conditional_label, np.tile(class_i, (int(class_i_num), 1))))
            num += class_i_num
        conditional_label = torch.FloatTensor(conditional_label)
        G_input = (torch.cat([conditional_label, torch.rand(points_num, self.z_dim)], dim=1))
        G_input = Variable(G_input.cuda())  # random ideas
        return G_input,conditional_label

    def E_training(self, points_num):
        num = 0
        for i in range(self.class_num):
            class_i = np.zeros((self.class_num))
            class_i[i] = 1

            class_i_num = round(points_num * (1 / float(self.class_num)))

            if i == self.class_num - 1:
                class_i_num = points_num - num
            if i == 0:
                conditional_label = np.tile(class_i, (int(class_i_num), 1))
            else:
                conditional_label = np.concatenate((conditional_label, np.tile(class_i, (int(class_i_num), 1))))

            num += class_i_num

        # E_NET:

        print ('training-E')

        for step in range(1000):

            G_input, conditional_label = self.gen_cond_label(points_num)
            self.G.eval()
            Generated_points = self.G(G_input,0)

            conditional_label=conditional_label.cpu().numpy()
            Generated_labels = Variable(torch.LongTensor(np.argmax(conditional_label, axis=1)).cuda())

            self.E_optimizer.zero_grad()
            pred = (self.E(Generated_points))
            #if step==0:
            #    print pred
                #for i in xrange(2000):
                #    print np.argmax(pred.data.cpu().numpy(), axis=1)[i],

            # print np.shape(Generated_labels)

            E_loss = torch.nn.CrossEntropyLoss()(pred, Generated_labels)
            E_loss.backward()
            self.E_optimizer.step()


    def entropy(self,y):
        y=nn.Softmax()(y)
        y1 = -y * torch.log(y + 1e-6)
        y2 = 1.0 / y1.size()[0] * y1.sum()

        return y2
    def data_preprocess(self):
        self.data_X, self.data_Y = utils.load_mnist(self.args.dataset)
        # .data_Y.size() (70000L, 10L)
        self.z_dim = 62
        self.y_dim = 10

        index = [0] * 10
        data = [0] * 10
        label = [0] * 10
        self.class_start = [0] * 10

        x = self.data_X
        _, y = torch.max(self.data_Y, 1)
        for i in range(10):
            index[i] = ((y == i).nonzero())
            data[i] = (x[torch.squeeze(index[i])])
            label[i] = torch.FloatTensor(np.tile(np.array(([i])), (data[i].size()[0], 1)))  # 1
            self.class_start[i] = label[i].size()[0]

        self.y = torch.cat(
            [label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9]], dim=0)
        self.x = (torch.cat([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]]))

        self.class_num = 10

        self.batch_size = 70
        torch_dataset = Data.TensorDataset(self.x, self.y)
        self.train_loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # random shuffle for training
            num_workers=2,  # subprocesses for loading data
        )
    def compute_accuracy(self):
        self.E.eval()
        end = 0
        correct = 0
        sample_num = 0
        for i in range(self.class_num):
            end_old = end
            end = end + self.class_start[i]
            y = self.E(Variable(self.x[end_old:end]).cuda())
            y = nn.Softmax()(y)
            sample_num = y.size()[0] + sample_num
            y = (y).data.cpu().numpy()  # [128,2]
            y = np.argmax(y, axis=1)
            counts = np.bincount(y)
            frequent = np.argmax(counts)
            correct += np.shape(((y == frequent).nonzero()))[1]
            print (y)
            print (frequent)
        print ('number of samples:', sample_num)
        accuracy = correct / float(sample_num)
        print ('accuracy:', accuracy)
        self.accuracy_hist.append(accuracy)
        plt.plot(self.accuracy_hist)
        plt.savefig("accuracy.png")
