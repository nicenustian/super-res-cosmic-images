import tensorflow as tf
tfkl = tf.keras.layers

###############################################################################
#WGAN CLASS for 1D data and 2D images
###############################################################################

class MSWGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        box_sizes,
        gp_weight=10.0,
    ):
        super(MSWGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.box_sizes = box_sizes
        self.gp_weight = gp_weight
        #self.d_steps = 4


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, 
                g_loss_fn, ps_loss_fn, pdf_loss_fn):
        
        super(MSWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.ps_loss_fn = ps_loss_fn
        self.pdf_loss_fn = pdf_loss_fn
        
                
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.p_loss_metric = tf.keras.metrics.Mean(name="p_loss")
        self.pdf_loss_metric = tf.keras.metrics.Mean(name="pdf_loss")
        self.total_g_loss_metric = tf.keras.metrics.Mean(name="total_g_loss")


    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.p_loss_metric]
    
    
    @tf.function
    def gradient_penalty(self, real_list, fake_list):
        
        avg_gp = 0.0
        batch_size = tf.shape(real_list[0])[0]
        
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)
        interpolated = [real + alpha * (fake - real) for real, fake in zip(real_list, fake_list)]
                
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
            
        grads = gp_tape.gradient(pred, interpolated)          
        
        #take average of gradients
        for grad in grads:
        
            #sum except the batch dims
            norm = tf.sqrt( tf.reduce_sum(tf.square(grad), [1,2,3]) )
            avg_gp += tf.reduce_mean((norm - 1.0) ** 2)

        return avg_gp/tf.cast(len(fake_list), tf.float32)
    
    
    #@tf.function
    def train_step(self, real_list):

        real = real_list
        box_sizes = self.box_sizes
        batch_size = tf.shape(real_list[0])[0]
        
        #CODE for extra steps for Discriminator training
        '''
        ############################DISCRIMNIMATOR###########################
        for i in range(self.d_steps):
            
            noise = tf.random.normal(shape=(self.batch_size, self.latent_dim)) \
                if self.latent else real_list[0]
            
            
            # all dis/gen pairs trained together with in scope of gradient tape
            with tf.GradientTape() as d_tape:
                                                    
                fake = self.generator(noise, training=True)
                fake_logits = self.discriminator(fake, training=True)
                real_logits = self.discriminator(real, training=True)
    
                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_logits, fake_logits)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(real, fake)
                d_loss =  d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            
                    
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables))
    
            
        with tf.GradientTape() as g_tape:
                        
            noise = tf.random.normal(shape=(self.batch_size, self.latent_dim)) \
                if self.latent else real_list[0]
            
            fake = self.generator(noise, training=True)
            fake_logits = self.discriminator(fake, training=True)
            g_loss = self.g_loss_fn(fake_logits)
            p_loss = self.ps_loss_fn(real, fake, box_sizes)
            pdf_loss = self.pdf_loss_fn(real, fake)
            
            total_g_loss = g_loss + p_loss + pdf_loss

        # Get the gradients w.r.t the generator loss
        g_gradient = g_tape.gradient(total_g_loss, self.generator.trainable_variables)

        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables))

        
    
        '''
        
        
        #CODE for combined Discriminator/genrator training
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # all dis/gen pairs trained together with in scope of gradient tape
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                                                
            fake = self.generator(noise, training=True)
            fake_logits = self.discriminator(fake, training=True)
            real_logits = self.discriminator(real, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_logits, fake_logits)

            # Calculate the gradient penalty
            gp = self.gradient_penalty(real, fake)
            
            p_loss = self.ps_loss_fn(real, fake, box_sizes)
            pdf_loss = self.pdf_loss_fn(real, fake)
            
            d_loss =  d_cost + gp * self.gp_weight + p_loss + pdf_loss
            g_loss = self.g_loss_fn(fake_logits) + p_loss + pdf_loss

           

        # Get the gradients w.r.t the dis/gen loss
        d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_variables)        
        g_gradient = g_tape.gradient(g_loss, self.generator.trainable_variables)

         
        # Update the weights of the discriminator using the dis.gen optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
            )

        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
            )
        

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.p_loss_metric.update_state(p_loss)
        self.pdf_loss_metric.update_state(pdf_loss)   
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "p_loss": self.p_loss_metric.result(),
            "pdf_loss": self.pdf_loss_metric.result(),
            "lr": self.d_optimizer.lr(self.d_optimizer.iterations)
        }

