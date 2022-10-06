import torch
from torch import nn
from .diffusion import GaussianDiffusion

class SDDM(nn.Module):
    def __init__(self, diffusion:GaussianDiffusion, noise_estimate_model:nn.Module,
                 noise_condition='sqrt_alpha_bar', p_transition='original', q_transition='original'):
        super().__init__()
        self.diffusion = diffusion
        self.noise_estimate_model = noise_estimate_model
        self.num_timesteps = self.diffusion.num_timesteps
        self.noise_condition = noise_condition
        self.p_transition = p_transition
        self.q_transition = q_transition
        if noise_condition != 'sqrt_alpha_bar' and noise_condition != 'time_step':
            raise NotImplementedError

        if p_transition != 'original' and p_transition != 'supportive' \
                and p_transition != 'sr3' and p_transition != 'conditional'\
                and p_transition != 'condition_in':
            raise NotImplementedError

        if q_transition != 'original' and q_transition != 'conditional':
            raise NotImplementedError

    # train step
    def forward(self, clean_audio, noisy_audio, noisy_spec):
        """
        clean_audio is the clean_audio source
        condition is the noisy conditional input
        """

        # generate noise
        if self.q_transition == 'original':
            noise = torch.randn_like(clean_audio, device=clean_audio.device)
            x_t, noise_level, t = self.diffusion.q_stochastic(clean_audio, noise)
            if self.noise_condition == 'sqrt_alpha_bar':
                predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, noise_level)
            elif self.noise_condition == 'time_step':
                predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, t)
            else:
                raise ValueError
        elif self.q_transition == 'conditional':
            noise = torch.randn_like(clean_audio, device=clean_audio.device)
            x_t, noise, t = self.diffusion.q_stochastic_conditional(clean_audio, noisy_audio, noise)
            predicted = self.noise_estimate_model(noisy_spec, x_t, t)
        else:
            raise ValueError

        return predicted, noise

    @torch.no_grad()
    def infer(self, noisy_audio, noisy_spec, continuous=False):
        # initial input

        # TODO: predict noise level to reduce computation cost

        if self.p_transition == 'conditional':
            # start from conditional input, conditional diffusion process
            x_t = self.diffusion.get_x_T_conditional(noisy_audio)
        elif self.p_transition == 'condition_in':
            # start from conditional input + gaussian noise, original diffusion process
            x_t = self.diffusion.get_x_T(noisy_audio)
        elif self.p_transition == 'supportive':
            # start from conditional input + gaussian noise, original diffusion process
            x_t = noisy_audio
        else:
            # start from total noise
            x_t = torch.randn_like(noisy_audio, device=noisy_audio.device)


        num_timesteps = self.diffusion.num_timesteps
        sample_inter = (1 | (num_timesteps // 100))

        batch_size = noisy_audio.shape[0]
        b = noisy_audio.shape[0]
        noise_level_sample_shape = torch.ones(noisy_audio.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b
        # iterative refinement

        samples = [noisy_audio]
        if continuous:
            assert batch_size==1, 'Batch size must be 1 to do continuous sampling'

        for t in reversed(range(1, self.num_timesteps+1)):

            if self.p_transition == 'original' or self.p_transition == 'condition_in':

                if self.noise_condition == 'sqrt_alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, time_steps)
                else:
                    raise ValueError

                x_t = self.diffusion.p_transition(x_t, t, predicted)
            elif self.p_transition == 'sr3':

                if self.noise_condition == 'sqrt_alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                    predicted = self.noise_estimate_model(x_t, noisy_audio, noisy_spec, time_steps)
                else:
                    raise ValueError

                x_t = self.diffusion.p_transition_sr3(x_t, t, predicted)
            elif self.p_transition == 'conditional':

                if self.noise_condition == 'sqrt_alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=noisy_audio.device)
                    predicted = self.noise_estimate_model(noisy_spec, x_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=noisy_audio.device)
                    predicted = self.noise_estimate_model(noisy_spec, x_t, time_steps)
                else:
                    raise ValueError
                x_t = self.diffusion.p_transition_conditional(x_t, t, predicted, noisy_audio)
            else:
                raise ValueError

            if continuous and t % sample_inter == 0:
                samples.append(x_t)

        if continuous:
            return samples
        else:
            return x_t


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())

        params = sum([p.numel() for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
