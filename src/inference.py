def predict(test_dir, model_path):
    model = load_model(model_path)
    test_images = load_test_images(test_dir)
    
    predictions = []
    for img in test_images:
        pred_mask = model(img)
        rle = mask_to_rle(pred_mask)
        predictions.append(rle)
    
    save_submission(predictions)