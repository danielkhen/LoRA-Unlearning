import torch

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            # Compute predictions
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            # Update the running total of correct predictions and samples
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_predictions

    model.train()

    return avg_loss, accuracy


def sample(model, num_classes, samples_per_class, input_size, timesteps, a_t, b_t, ab_t):
    # Set model to evaluation mode
    model.eval()

    # Start from pure noise
    labels = torch.nn.functional.one_hot(torch.arange(0, num_classes, dtype=torch.long).repeat_interleave(samples_per_class))
    labels = torch.cat((labels, torch.zeros(samples_per_class, num_classes))).to('cuda');
    x = torch.randn(labels.shape[0], 1, input_size, input_size).to('cuda')

    for t in reversed(range(1, timesteps + 1)):
        t_tensor = torch.full((labels.shape[0],), t, device='cuda', dtype=torch.long)
        t_norm = t_tensor / timesteps

        # Predict noise
        pred_noise = model(x, t_norm, labels)

        # Compute coefficients
        beta_t = b_t[t].view(-1, 1, 1, 1)
        alpha_t = a_t[t].view(-1, 1, 1, 1)
        ab_t_ = ab_t[t].view(-1, 1, 1, 1)

        # DDPM sampling step
        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - ab_t_).sqrt()) * pred_noise) + beta_t.sqrt() * noise

    # Convert to numpy and denormalize for visualization
    samples = x.clamp(-1, 1).cpu().numpy()
    samples = (samples * 0.5) + 0.5  # if your data was normalized to [-1, 1]

    return samples