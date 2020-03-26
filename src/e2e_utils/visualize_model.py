# Local libraries
from e2e_utils.show_model_samples import show_image

# External libraries
import matplotlib.pyplot as plt
import torch



def visualize_model(model, plot_name, class_names, device, data_loaders,
                    num_images=6):
    """
    Visualize model
    """
    was_training = model.training
    model.eval()
    images_so_far = 0

    # Initialize plt canvas
    fig = plt.figure()

    # Loading model
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders['finetune']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            try:
                outputs,aux = model(inputs)
            except:
                outputs = model(inputs)
                
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                # Save image
                show_image(data_loaders['finetune'], plot_prefix=plot_name,
                           class_names=class_names, save=False)

                #print("true label: {}".format(class_names[labels.cpu()[j]]))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    fig_name = f'{plot_name}_true_labels_norm.png'
                    plt.savefig(fig_name, bbox_inches="tight")
                    return

        model.train(mode=was_training)