import datetime

import matplotlib.pyplot as plt
import numpy as np

import torch


def visualize_loss(train_losses, test_losses, title):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d")
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Train and Test Losses\n Ritarka: {timestamp_str}")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


def visualize_images(model, testset, device, title):
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d")

    # Visualize the final results
    plt.figure(figsize=(12, 6))
    random_inds = np.random.choice(len(testset), 10)

    for i, image_idx in enumerate(random_inds):
        test_image, test_label = testset[image_idx]
        test_image = test_image.unsqueeze(0)

        # Move the test_image tensor to the same device as the model (cuda or cpu)
        test_image = test_image.to(device)

        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # Move the test_image tensor back to CPU for visualization
        test_image_cpu = test_image.cpu().squeeze().numpy()
        if len(test_image_cpu.shape) == 3:
            test_image_cpu = (test_image_cpu.transpose(1, 2, 0) + 1) / 2
        plt.imshow(test_image_cpu, cmap=plt.cm.binary)

        with torch.no_grad():
            output = model(test_image)
            predicted_label = torch.argmax(output).item()

        plt.xlabel(f"True: {test_label}\nPred: {predicted_label}")

    plt.suptitle(
        f"Final Results - {title}\n Ritarka: {timestamp_str}",
        fontsize=16,
    )
    plt.show()