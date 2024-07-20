import yaml

import wandb
from imgalaxy.cfg import PKG_PATH
from imgalaxy.constants import IMAGE_SIZE, NUM_EPOCHS, THRESHOLD
# from imgalaxy.helpers import check_augmented_images, evaluate_model
from imgalaxy.unet import UNet


def train(config):
   
        unet = UNet(
            loss=config.loss,
            dropout_rate=config.dropout_rate,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            batch_normalization=config.batch_normalization,
            kernel_regularization=config.kernel_regularization,
            image_size=config.image_size,
            n_filters=config.n_filters,
            mask=config.mask,
            min_vote=config.min_vote,
        )

        # wandb.log(
        #      dict(
        #         t_loss=unet.loss,
        #         t_dropout_rate=unet.dropout_rate,
        #         t_num_epochs=unet.num_epochs,
        #         t_learning_rate=unet.learning_rate,
        #         t_batch_size=unet.batch_size,
        #         t_batch_normalization=unet.batch_normalization,
        #         t_kernel_regularization=unet.kernel_regularization,
        #         t_image_size=unet.image_size,
        #         t_n_filters=unet.n_filters,
        #         t_mask=unet.mask,
        #         t_min_vote=unet.min_vote,  
        #      )
        # )
        _, _, _ = unet.train_pipeline()
        # check_augmented_images(train_data)
        # evaluate_model(test_data, unet.unet_model, num=3)

def main():
    wandb.init(project="galaxy-segmentation-project")
    train(wandb.config)


if __name__ == '__main__':
    sweep_configs = yaml.safe_load((PKG_PATH / 'sweep.yaml').read_text())
    sweep_id = wandb.sweep(sweep=sweep_configs, project="galaxy-segmentation-project")
    wandb.agent(sweep_id, function=main)
    # wandb.agent(
    #     f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=47
    # )
    #train()
