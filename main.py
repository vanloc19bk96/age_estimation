import torch
import logging
import data_preparetion

data_path = "data"
def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s")

    # set cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = data_preparetion.prepare_data(data_path)

    best_MAEs = []
    last_MAEs = []

    for id in range(len(dataset['train'])):
        # init model
        model = model_preparetion.prepare_model()

if __name__ == '__main__':
    main()


