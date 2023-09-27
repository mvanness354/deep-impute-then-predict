import tensorflow as tf
from tensorflow_probability import distributions as tfd

class MLPBlock(tf.keras.Model):
    def __init__(
        self,
        d_hidden=128,
        n_layers=2,
        activation=tf.nn.relu,
        dropout=0.0,
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(tf.keras.layers.Dense(d_hidden, activation=activation))
                  
        layers.append(tf.keras.layers.Dropout(dropout))
        self.mlp = tf.keras.Sequential(layers)

    def call(self, x):
        # mask = tf.cast(tf.math.is_nan(x), tf.float32)
        # x = tf.where(tf.math.is_nan(x), 0.0, x)

        h = self.mlp(x)

        return h

class MLPRegressor(tf.keras.Model):
    def __init__(
        self,
        d_hidden=128,
        n_layers=2,
        activation=tf.nn.relu,
        dropout=0.0,
        probabilistic=False,
        add_mask=False,
    ):
        super().__init__()
        
        self.probabilistic = probabilistic
        self.add_mask = add_mask

        layers = []
        for _ in range(n_layers):
            layers.append(tf.keras.layers.Dense(d_hidden, activation=activation))
                
        layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(1, activation=None))
        self.mlp = tf.keras.Sequential(layers)

    def impute(self, x):
        return tf.where(tf.math.is_nan(x), 0.0, x)

    def call(self, x):
        
        if self.add_mask:
            mask = tf.cast(tf.math.is_nan(x), tf.float32)
            x = tf.where(tf.math.is_nan(x), 0.0, x)
            x = tf.concat([x, mask], axis=1)
        else:
            x = tf.where(tf.math.is_nan(x), 0.0, x)

        yhat = tf.squeeze(self.mlp(x), -1)
        
        if self.probabilistic:
            return tfd.Normal(loc=yhat, scale=tf.ones_like(yhat))
        else:
            return yhat
        
class MLPClassifier(tf.keras.Model):
    def __init__(
        self,
        n_classes=2,
        d_hidden=128,
        n_layers=2,
        activation=tf.nn.relu,
        dropout=0.0,
        probabilistic=False,
        add_mask=False,
    ):
        super().__init__()
        
        self.probabilistic = probabilistic
        self.add_mask = add_mask

        self.n_classes = n_classes
        output_shape = n_classes if n_classes > 2 else 1

        layers = []
        for _ in range(n_layers):
            layers.append(tf.keras.layers.Dense(d_hidden, activation=activation))
                
        layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(output_shape, activation=None))
        self.mlp = tf.keras.Sequential(layers)

    def impute(self, x):
        return tf.where(tf.math.is_nan(x), 0.0, x)

    def call(self, x):

        if self.add_mask:
            mask = tf.cast(tf.math.is_nan(x), tf.float32)
            x = tf.where(tf.math.is_nan(x), 0.0, x)
            x = tf.concat([x, mask], axis=1)
        else:
            x = tf.where(tf.math.is_nan(x), 0.0, x)

        logits = self.mlp(x)
        
        if self.n_classes == 2:
            probs = tf.keras.activations.sigmoid(tf.squeeze(logits, -1))
        else:
            probs = tf.keras.activations.softmax(logits)

            
        probs = tf.clip_by_value(probs, clip_value_min=0.001, clip_value_max=0.999)
        
        if self.probabilistic:
            if self.n_classes == 2:
                return tfd.Bernoulli(probs=probs)
            else:
                return tfd.Categorical(probs=probs)
        else:
            return probs
        
class MLPMultiLabel(tf.keras.Model):
    def __init__(
        self,
        n_out,
        d_hidden=128,
        n_layers=2,
        activation=tf.nn.relu,
        dropout=0.0,
        probabilistic=False,
        add_mask=False,
        seperate_heads=False
    ):
        super().__init__()

        self.probabilistic = probabilistic
        self.add_mask = add_mask
        self.seperate_heads = seperate_heads

        if self.seperate_heads:
            self.mlps = [
                MLPClassifier(
                    n_classes=2, 
                    d_hidden=d_hidden, 
                    n_layers=n_layers, 
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(n_out)
            ]
        else:
            self.mlp = MLPClassifier(
                n_classes=n_out,
                d_hidden=d_hidden,
                n_layers=n_layers,
                activation=activation,
                dropout=dropout,
            )

    def impute(self, x):
        return tf.where(tf.math.is_nan(x), 0.0, x)

    def call(self, x):

        if self.add_mask:
            mask = tf.cast(tf.math.is_nan(x), tf.float32)
            x = tf.where(tf.math.is_nan(x), 0.0, x)
            x = tf.concat([x, mask], axis=1)
        else:
            x = tf.where(tf.math.is_nan(x), 0.0, x)
        
        if self.seperate_heads:
            probs = tf.stack([
                mlp(x) for mlp in self.mlps
            ], axis=-1)
        else:
            probs = self.mlp(x)
        
        probs = tf.clip_by_value(probs, clip_value_min=0.001, clip_value_max=0.999)
        
        if self.probabilistic:
            return tfd.Bernoulli(
                probs=probs
            )
        else:
            return probs
        
    
    
class NeuMissBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_features,
        depth = 3,
        impute=False
    ):
        super().__init__()
        
        self.depth = depth
        self.impute = impute
        self.mu = tf.Variable(tf.random.normal([n_features]))
        self.Wobs = tf.keras.layers.Dense(n_features, use_bias=False)

        if impute:
            self.Wmix = tf.keras.layers.Dense(n_features, use_bias=False)
        
    def call(self, x):
        
        mask = tf.cast(tf.math.is_nan(x), tf.float32)
        x = tf.where(tf.math.is_nan(x), 0.0, x)
        h = x - (1.0 - mask)*self.mu
        skip = h
        
        for _ in range(self.depth):
            h = (1.0 - mask)*self.Wobs(h)
            h += skip

        if self.impute:
            # impute missing entries and add back in observe entries
            h = (self.Wmix(h) + self.mu) * mask + x
            
        return h
        
    
class NeuMissMLP(tf.keras.Model):
    def __init__(
        self,
        n_input,
        n_classes=2,
        neumiss_depth=3,
        mlp_d_hidden=128,
        mlp_n_layers=2,
        activation=tf.nn.relu,
        dropout=0.0,
        multilabel=False,
        regression=False,
        n_out=None,
        add_mask=False,
        impute=False
    ):
        super().__init__()
        
        if multilabel:
            self.mlp = MLPMultiLabel(
                n_out=n_out,
                d_hidden=mlp_d_hidden, 
                n_layers=mlp_n_layers,
                activation=activation, 
                dropout=dropout, 
            )
        elif regression:
            self.mlp = MLPRegressor(
                d_hidden=mlp_d_hidden,
                n_layers=mlp_n_layers,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.mlp = MLPClassifier(
                n_classes=n_classes,
                d_hidden=mlp_d_hidden,
                n_layers=mlp_n_layers,
                activation=activation,
                dropout=dropout,
            )
        
        self.to_impute = impute
        self.neumiss_block = NeuMissBlock(n_input, depth=neumiss_depth, impute=impute)
        self.add_mask = add_mask

    def impute(self, x):
        if self.to_impute:
            return self.neumiss_block(x)
        else:
            raise ValueError("Cannot impute when impute=False")
        
    def call(self, x):

        h = self.neumiss_block(x)
        # self.imputations = h.copy().numpy()
        # self.inputs = x.copy.numpy()

        if self.add_mask:
            mask = tf.cast(tf.math.is_nan(x), tf.float32)
            h = tf.concat([h, mask], axis=1)

        return self.mlp(h)
    
    
class VAEBlock(tf.keras.Model):
    def __init__(self,
                 n_out,
                 d_hidden=128,
                 n_layers=2,
                 activation=tf.nn.relu,
                 name_="encoder"
                ):
        super().__init__()
        
        self.name_ = name_
        self.mlp = MLPBlock(d_hidden=d_hidden, n_layers=n_layers, activation=activation)
        self.loc_head = tf.keras.layers.Dense(n_out)
        self.logscale_head = tf.keras.layers.Dense(n_out)
        
    def check_nan(self, x, name):
        # is_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.float32))
        # _ = tf.cond(tf.greater(is_nan, 0), lambda: print(f"\nNAN in {name}"), lambda: None)
        # return
        pass

    def call(self, x):
        
        self.check_nan(x, f"{self.name_} X")
        
        h = self.mlp(x)
        
        self.check_nan(h, f"{self.name_} H")
        
        loc = self.loc_head(h)
        log_scale = self.logscale_head(h)
        
        self.check_nan(loc, f"{self.name_} LOC")
        self.check_nan(log_scale, f"{self.name_} LOG_SCALE")
        self.check_nan(tf.nn.softplus(log_scale), f"{self.name_} LOG SCALE SOFTPLUS")

        return tfd.Normal(loc, tf.nn.softplus(log_scale) + 1e-6)
    
    
class MLPMIWAE(tf.keras.Model):
    
    def __init__(self,
                 n_input,
                 n_classes=2,
                 d_hidden=128,
                 n_layers=2,
                 activation=tf.nn.elu,
                 n_samples=2,
                 dropout=0.0,
                 multilabel=False,
                 regression=False,
                 n_out=None,
                ):

        super().__init__()
        
        self.K = n_samples
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.regression = regression
        
        self.encoder = VAEBlock(n_out=n_input, d_hidden=d_hidden, n_layers=n_layers, activation=activation, name_="encoder")
        self.decoder = VAEBlock(n_out=n_input, d_hidden=d_hidden, n_layers=n_layers, activation=activation, name_="decoder")
        if multilabel:
            self.disc = MLPMultiLabel(
                n_out=n_out, 
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
                probabilistic=True
            )
        elif regression:
            self.disc = MLPRegressor(
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
                probabilistic=True
            )
        else:
            self.disc = MLPClassifier(
                n_classes=n_classes, 
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
                probabilistic=True,
            )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def logmeanexp(self, log_w, axis):
        max_ = tf.reduce_max(log_w, axis=axis, keepdims=True)
        
        exp_ = tf.exp(log_w - max_)
        
        return tf.squeeze(tf.math.log(tf.reduce_mean(
            exp_, axis=axis, keepdims=True
        )) + max_, axis)
    
    def check_nan(self, x, name):
        # is_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.float32))
        # _ = tf.cond(tf.greater(is_nan, 0), lambda: print(f"\nNAN in {name}"), lambda: None)
        # return
        pass
        
    def supmiwae_loss(self, y, outputs):
        lpyx = outputs['pyx'].log_prob(y)
        
        if self.multilabel:
            lpyx = tf.reduce_sum(lpyx, axis=-1)
        
        log_w = lpyx + outputs['lpxz'] + outputs['lpz'] - outputs['lqzx']
                
        self.check_nan(lpyx, "LPYX")
        self.check_nan(outputs['lpxz'], "LPXZ")
        self.check_nan(outputs['lpz'], "LPZ")
        self.check_nan(outputs['lqzx'], "LQZX")
        
        elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)
        return -elbo

    def impute(self, x, reduce=True):
        mask = tf.cast(tf.math.is_nan(x), tf.float32)
        x = tf.where(tf.math.is_nan(x), 0.0, x)
        
        self.check_nan(x, "X")

        # ---- prior p(z)
        pz = tfd.Normal(0, 1)

        # ---- variational posterior q(z|x)
        qzx = self.encoder(x)

        z = qzx.sample(self.K)
    
        self.check_nan(z, "Z")

        # ---- observation model p(x|z)
        pxz = self.decoder(z)

        # ---- samples from the observation model
        x_samples = pxz.sample()

        if reduce:
            # ----------- log probs
            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)
            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)
            lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)
            
            # -------- y probs use self-normalized importance weights
            iwae_log_w = lpxz + lpz - lqzx
            snis = tf.math.softmax(iwae_log_w, axis=0)

            x_samples = tf.reduce_sum(snis[:, :, None] * x_samples, axis=0)

            # ---- mix observed data with samples of the missing data
            x_mixed = (1 - mask) * x + mask * x_samples

        else:
            # ---- mix observed data with samples of the missing data
            x_mixed = (1 - mask[None, :, :]) * x[None, :, :] + mask[None, :, :] * x_samples

        return x_mixed

    def call(self, x):  

        mask = tf.cast(tf.math.is_nan(x), tf.float32)
        x = tf.where(tf.math.is_nan(x), 0.0, x)
        
        self.check_nan(x, "X")

        # ---- prior p(z)
        pz = tfd.Normal(0, 1)

        # ---- variational posterior q(z|x)
        qzx = self.encoder(x)

        z = qzx.sample(self.K)
        
        self.check_nan(z, "Z")

        # ---- observation model p(x|z)
        pxz = self.decoder(z)

        # ---- samples from the observation model
        x_samples = pxz.sample()
        
        self.check_nan(x_samples, "X SAMPLES")

        # ---- mix observed data with samples of the missing data
        x_mixed = (1 - mask[None, :, :]) * x[None, :, :] + mask[None, :, :] * x_samples

        # ---- discriminator p(y|x)
        pyx = self.disc(x_mixed)
        
        # ----------- log probs
        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)
        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)
        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)
        
        # -------- y probs use self-normalized importance weights
        iwae_log_w = lpxz + lpz - lqzx
        snis = tf.math.softmax(iwae_log_w, axis=0)
        
        if self.regression:
            y_sample_preds = pyx.loc
        else:
            y_sample_preds = pyx.probs
        
        if self.n_classes != 2 or self.multilabel:
            y_preds = tf.reduce_sum(snis[:, :, None] * y_sample_preds, axis=0)
        else:
            y_preds = tf.reduce_sum(snis[:, :] * y_sample_preds, axis=0)
            
        self.check_nan(y_preds, "Y PROBS")

        return y_preds, {'lpz': lpz, 'lqzx': lqzx, 'lpxz': lpxz, 'pyx': pyx}

    def train_step(self, data):

        # data = data_adapter.expand_1d(data)
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # tf.print(x.shape)
        
        with tf.GradientTape() as tape:
            y_pred, state = self(x, training=True)
            loss_value = self.supmiwae_loss(y, state)

        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss_value)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        
        output = {}
        output["loss"] = self.loss_tracker.result()
        for m in self.metrics:
            output[m.name] = m.result()
        return output
        
    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred, state = self(x, training=False)
        
        loss_value = self.supmiwae_loss(y, state)
        self.loss_tracker.update_state(loss_value)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        
        output = {}
        output["loss"] = self.loss_tracker.result()
        for m in self.metrics:
            output[m.name] = m.result()
        return output
    
# Combination of Not-MIWAE and Sup-MIWAE
class MLPNotMIWAE(tf.keras.Model):
    
    def __init__(self,
                 n_input,
                 n_classes=2,
                 d_hidden=128,
                 n_layers=2,
                 activation=tf.nn.elu,
                 n_samples=2,
                 dropout=0.0,
                 multilabel=False,
                 regression=False,
                 n_out=None,
                 **kwargs):

        super().__init__()
        
        self.K = n_samples
        self.n_classes = n_classes
        self.multilabel = multilabel
        self.regression = regression
        
        self.encoder = VAEBlock(n_out=n_input, d_hidden=d_hidden, n_layers=n_layers)
        self.decoder = VAEBlock(n_out=n_input, d_hidden=d_hidden, n_layers=n_layers)
        
        if multilabel:
            self.disc = MLPMultiLabel(
                n_out=n_out,
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
                probabilistic=True
            )
        elif regression:
            self.disc = MLPRegressor(
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
                probabilistic=True
            )
        else:
            self.disc = self.disc = MLPClassifier(
                n_classes=n_classes, 
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
                probabilistic=True
            )
        self.mask_pred = MLPMultiLabel(
            n_out=n_input, 
            d_hidden=d_hidden, 
            n_layers=n_layers,
            activation=activation, 
            dropout=dropout, 
            probabilistic=True
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def logmeanexp(self, log_w, axis):
        max = tf.reduce_max(log_w, axis=axis, keepdims=True)
        return tf.squeeze(tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis, keepdims=True)) + max, axis)
        
    def supmiwae_loss(self, y, outputs):
        lpyxs = outputs['pyxs'].log_prob(y)
        
        if self.multilabel:
            lpyxs = tf.reduce_sum(lpyxs, axis=-1)
        
        log_w = lpyxs + outputs['lpsx'] + outputs['lpxz'] + outputs['lpz'] - outputs['lqzx']
        elbo = tf.reduce_mean(self.logmeanexp(log_w, axis=0), axis=-1)
        # print("\nELBO", elbo.numpy())
        return -elbo
    
    def check_nan(self, x, name):
        # is_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.float32))
        # _ = tf.cond(tf.greater(is_nan, 0), lambda: print(f"\nNAN in {name}"), lambda: None)
        # return
        pass

    def impute(self, x, reduce=True):
        mask = tf.cast(tf.math.is_nan(x), tf.float32)
        x = tf.where(tf.math.is_nan(x), 0.0, x)
        
        self.check_nan(x, "X")

        # ---- prior p(z)
        pz = tfd.Normal(0, 1)

        # ---- variational posterior q(z|x)
        qzx = self.encoder(x)

        z = qzx.sample(self.K)
        
        self.check_nan(z, "Z")

        # ---- observation model p(x|z)
        pxz = self.decoder(z)

        # ---- samples from the observation model
        x_samples = pxz.sample()

        # ---- mix observed data with samples of the missing data
        x_mixed = (1 - mask[None, :, :]) * x[None, :, :] + mask[None, :, :] * x_samples

        # ------- mask predictor p(s|x)
        psx = self.mask_pred(x_mixed)

        if reduce:
            # ----------- log probs
            lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)
            lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)
            lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)
            lpsx = tf.reduce_sum(psx.log_prob(mask), axis=-1)
            
            # -------- y probs use self-normalized importance weights
            iwae_log_w = lpsx + lpxz + lpz - lqzx
            snis = tf.math.softmax(iwae_log_w, axis=0)

            x_samples = tf.reduce_sum(snis[:, :, None] * x_samples, axis=0)

            # ---- mix observed data with samples of the missing data
            x_mixed = (1 - mask) * x + mask * x_samples

        else:
            # ---- mix observed data with samples of the missing data
            x_mixed = (1 - mask[None, :, :]) * x[None, :, :] + mask[None, :, :] * x_samples

        return x_mixed

    def call(self, x):
        
        mask = tf.cast(tf.math.is_nan(x), tf.float32)
        x = tf.where(tf.math.is_nan(x), 0.0, x)

        # ---- prior p(z)
        pz = tfd.Normal(0, 1)

        # ---- variational posterior q(z|x)
        qzx = self.encoder(x)

        z = qzx.sample(self.K)
        
        
        # ---- observation model p(x|z)
        pxz = self.decoder(z)

        # ---- samples from the observation model
        x_samples = pxz.sample()

        # ---- mix observed data with samples of the missing data
        x_mixed = (1 - mask[None, :, :]) * x[None, :, :] + mask[None, :, :] * x_samples

        # ---- discriminator p(y|x,s)
        disc_input = tf.concat(
            [x_mixed,
            tf.tile(tf.expand_dims(mask, 0), [x_mixed.shape[0], 1, 1])],
            axis=-1
        )
        pyxs = self.disc(disc_input)
        
        # ------- mask predictor p(s|x)
        psx = self.mask_pred(x_mixed)
        
        # ----------- log probs
        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)
        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)
        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)
        lpsx = tf.reduce_sum(psx.log_prob(mask), axis=-1)
        
        # -------- y probs use self-normalized importance weights
        iwae_log_w = lpsx + lpxz + lpz - lqzx
        snis = tf.math.softmax(iwae_log_w, axis=0)
        
        if self.regression:
            y_sample_preds = pyxs.loc
        else:
            y_sample_preds = pyxs.probs
        
        if self.n_classes != 2 or self.multilabel:
            y_preds = tf.reduce_sum(snis[:, :, None] * y_sample_preds, axis=0)
        else:
            y_preds = tf.reduce_sum(snis[:, :] * y_sample_preds, axis=0)
            
        self.check_nan(y_preds, "Y PROBS")

        return y_preds, {'lpz': lpz, 'lqzx': lqzx, 'lpxz': lpxz, 'lpsx': lpsx, 'pyxs': pyxs}

    def train_step(self, data):
        # data = data_adapter.expand_1d(data)
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # x, mask, y = data
        # tf.print(x.shape)
        
        
        with tf.GradientTape() as tape:
            y_pred, state = self(x, training=True)
            loss_value = self.supmiwae_loss(y, state)

        grads = tape.gradient(loss_value, self.trainable_variables)
        # for grad, var in list(zip(grads, self.trainable_variables))[:5]:
        #     print(var.name, tf.reduce_max(grad).numpy())
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss_value)
        self.compiled_metrics.update_state(y, y_pred)
        
        output = {}
        output["loss"] = self.loss_tracker.result()
        for m in self.metrics:
            output[m.name] = m.result()
        return output
        
    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred, state = self(x, training=False)
        
        loss_value = self.supmiwae_loss(y, state)
        self.loss_tracker.update_state(loss_value)
        self.compiled_metrics.update_state(y, y_pred)
        
        output = {}
        output["loss"] = self.loss_tracker.result()
        for m in self.metrics:
            output[m.name] = m.result()
        return output
        

class AutoEncoder(tf.keras.Model):
    def __init__(
        self, 
        theta, 
        n_layers=4, 
        d_hidden=128, 
        activation=tf.nn.relu
    ):

        super().__init__()
        self.theta = theta

        encoder_layers = []
        for i in range(1, n_layers):
            encoder_layers.append(tf.keras.layers.Dense(d_hidden + theta*i, activation=activation))
        encoder_layers.append(tf.keras.layers.Dense(d_hidden + theta*n_layers, activation=None))         
        self.encoder = tf.keras.Sequential(encoder_layers)

        decoder_layers = []
        for i in reversed(range(1, n_layers)):
            decoder_layers.append(tf.keras.layers.Dense(d_hidden + theta*i, activation=activation))
        decoder_layers.append(tf.keras.layers.Dense(d_hidden, activation=None))         
        self.decoder = tf.keras.Sequential(decoder_layers)

    def call(self, x):
        x = tf.where(tf.math.is_nan(x), 0.0, x)
        
        return self.decoder(self.encoder(x))
    
class AutoEncodePredictor(tf.keras.Model):

    def __init__(self,
                 n_input,
                 n_classes=2,
                 d_hidden=128,
                 n_layers=2,
                 activation=tf.nn.relu,
                 regression=False,
                 add_mask=False,
                 dropout=0.0,
                 theta=7,
    ):

        super().__init__()
        self.ae = AutoEncoder(n_layers=n_layers, d_hidden=n_input, activation=activation, theta=theta)
        self.add_mask = add_mask

        if regression:
            self.output_head = MLPRegressor(
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
            )
        else:
            self.output_head = self.disc = MLPClassifier(
                n_classes=n_classes, 
                d_hidden=d_hidden, 
                n_layers=n_layers,
                activation=activation, 
                dropout=dropout, 
            )

    def impute(self, x):
        return self.ae(x)
    
    def call(self, x):
        mask = tf.cast(tf.math.is_nan(x), tf.float32)
        h = self.ae(x)

        if self.add_mask:
            h = tf.concat([h, mask], axis=1)

        return self.output_head(h)


