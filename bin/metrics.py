def _cal_loss(self, lstm_outputs):
    batch_size = self.options['batch_size']
    unroll_steps = self.options['unroll_steps']

    n_tokens_vocab = self.options['n_tokens_vocab']
    softmax_dim = self.options['lstm']['projection_dim']    # 512
    