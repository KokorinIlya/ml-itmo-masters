from resnet import ResNet
from cifar import load_cifar10
from train import train_model
import torch
import argparse
from evaluation import calc_accuracy


def str2bool(s):
    assert s in {'True', 'False'}
    return True if s == 'True' else False


def save_model(model, accs, model_name, continue_train):
    torch.save(model.state_dict(), 'models/{0}.pt'.format(model_name))
    write_mode = 'a' if continue_train else 'w'
    with open('logs/{0}.log'.format(model_name), write_mode) as f:
        for acc in accs:
            f.write('{0}\n'.format(acc))


def load_model(model_name, layers_count):
    model = ResNet(n=layers_count)
    teacher_weights = torch.load('models/{0}.pt'.format(model_name))
    model.load_state_dict(teacher_weights)
    return model


def load_epoch_passed(model_name):
    with open('logs/{0}.log'.format(model_name), 'r') as f:
        lines = f.readlines()
        non_empty_lines = list(filter(lambda x: x.strip() != '', lines))
        assert len(non_empty_lines) >= 1
        return len(non_empty_lines) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help='Mode of execution. Can be either:\n'
                                                      'train - train the new model from scratch\n'
                                                      'continue - continue training of the existing model\n'
                                                      'eval - evaluate the accuracy of the existing model')
    parser.add_argument("--layers", required=True, type=int,
                        help='Number of layers in the model, being trained or evaluated')
    parser.add_argument("--name", required=True, help='Name of the model, being trained or evaluated')

    parser.add_argument("--epochs", required=False, type=int,
                        help='Number of epochs to train. Required if mode is train or continue')
    parser.add_argument("--use_teacher", required=False, type=str2bool,
                        help='Determines, whether a teacher should be used for training or not. '
                             'Required if mode is train or continue')
    parser.add_argument("--teacher_name", required=False,
                        help='Name of the teacher mode. Required if use_teacher is set to True')
    parser.add_argument("--teacher_layers", required=False, type=int,
                        help='Number of layers in the teacher model. Required if use_teacher is set to True')
    args = parser.parse_args()
    assert args.mode in {'train', 'continue', 'eval'}

    test_dataset = load_cifar10(is_train=False, save_path='data')

    if args.mode == 'eval':
        model = load_model(model_name=args.name, layers_count=args.layers)
        acc = calc_accuracy(model=model, test_dataset=test_dataset)
        print('Accuracy = {0}'.format(acc))
    else:
        assert args.mode in {'train', 'continue'}
        train_dataset = load_cifar10(is_train=True, save_path='data')
        if args.mode == 'continue':
            model = load_model(model_name=args.name, layers_count=args.layers)
            epochs_passed = load_epoch_passed(model_name=args.name)
        else:
            model = ResNet(n=args.layers)
            epochs_passed = 0

        if not args.use_teacher:
            accs = train_model(model=model, epochs=args.epochs,
                               train_dataset=train_dataset, test_dataset=test_dataset,
                               epochs_passed=epochs_passed)
        else:
            teacher_model = load_model(model_name=args.teacher_name, layers_count=args.teacher_layers)
            accs = train_model(model=model, epochs=args.epochs,
                               teacher=teacher_model, alpha=0.5,
                               train_dataset=train_dataset, test_dataset=test_dataset,
                               epochs_passed=epochs_passed)
        save_model(model=model, accs=accs, model_name=args.name, continue_train=args.mode == 'continue')


if __name__ == '__main__':
    main()
