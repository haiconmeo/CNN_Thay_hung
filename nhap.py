from tensorflow.python.keras.preprocessing import image

ef generator(df, batch_size):
    img_list = df['img']
    wheel_axis = df['wheel-axis']    
    # create an empty batch
    batch_img = np.zeros((batch_size,) + img_shape)
    batch_label = np.zeros((batch_size, 1))
    index = 0
    while True:
        for i in range(batch_size):
            label = wheel_axis.iloc[index]
            img_name = img_list.iloc[index]
            pil_img = image.load_img(path_to_data+img_name)
            # Data augmentation           
            if(np.random.choice(2, 1)[0] == 1):
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                label = -1 * label            
            batch_img[i] = image.img_to_array(pil_img)
            batch_label[i] = label
            index += 1
            if index == len(img_list):
                #End of an epoch hence reshuffle
                df = df.sample(frac=1).reset_index(drop=True)
                img_list = df['img']
                wheel_axis = df['wheel-axis']
                index = 0
        yield batch_img / INPUT_NORMALIZATION, (batch_label / OUTPUT_NORMALIZATION)