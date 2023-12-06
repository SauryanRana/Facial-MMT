import numpy as np
from sklearn.metrics import f1_score

# def eval_affwild(preds, label_orig):
#     val_preds = preds.cpu().detach().numpy()
#     val_true = label_orig.cpu().detach().numpy() 
#     predicted_label = []
#     true_label = []
#     for i in range(val_preds.shape[0]):
#         predicted_label.append(np.argmax(val_preds[i,:],axis=0) ) #
#         true_label.append(val_true[i])
#     macro_av_f1 = f1_score(true_label, predicted_label, average='macro')
#     return macro_av_f1


def eval_meld(results, truths, test=False):
    print("inside eval_meld")
    test_preds = results.cpu().detach().numpy()   #（num_utterance, num_label)
    print("test_preds: " , test_preds)
    test_truth = truths.cpu().detach().numpy()  #（num_utterance）
    print("test_truth: ",test_truth)
    predicted_label = []
    true_label = []
    
    print("test_preds.shape[0]: ", test_preds.shape[0])
    for i in range(test_preds.shape[0]):
        print("i", i)
        print(test_preds[i,:])
        print(" (np.argmax(test_preds[i,:],axis=0)" , np.argmax(test_preds[i,:],axis=0))
        predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) 
        true_label.append(test_truth[i])
    wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
    print("wg_av_f1: ", wg_av_f1)
    
    if test:
        print("inside if test: ")
        print("true label: ", true_label)
        print("predicted_label: ", predicted_label)
        f1_each_label = f1_score(true_label, predicted_label, average=None)    
        print('**TEST** | f1 on each class (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger): \n', f1_each_label)
        
        
    print("OUR FUNCTION IS RETURNING: ", wg_av_f1)
    return wg_av_f1

