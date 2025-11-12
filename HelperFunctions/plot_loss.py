import matplotlib.pyplot as plt
def plot_loss(loss1:list,epochs:int,loss2:list=None,label1:str=None,label2:str=None)->None:
    '''
    Function for plotting losses
    :param loss1: List of loss per epoch
    :param epochs: Number of cycles
    :param loss2: List of loss per epoch
    :param label1: Label for loss1
    :param label2: Label for loss2
    :return: None
    '''
    plt.plot(range(1,epochs+1),loss1,label=label1)
    if loss2 is not None:
        plt.plot(range(1,epochs+1),loss2,label=label2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss Plot')
    plt.legend()
    plt.show()