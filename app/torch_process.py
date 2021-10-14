import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,DataCollatorWithPadding, AdamW, get_scheduler,get_linear_schedule_with_warmup,)

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


class torch_pred():
    def __init__(self,df):
        checkpoint = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        tweets = df

        checkpoint = "distilbert-base-uncased"
        PATH = "app/toxic_distilBERT_multilabel_save.pt"
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 6)
        self.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        self.device = torch.device("cpu")



        sub_tokens = tokenizer.batch_encode_plus(tweets["cleaned_tweets"].tolist(),
                                         max_length = 200,
                                         pad_to_max_length=True,
                                         truncation=True,
                                         return_token_type_ids=False
                                         )

        sub_seq = torch.tensor(sub_tokens['input_ids'])
        sub_mask = torch.tensor(sub_tokens['attention_mask'])
        self.sub_data = TensorDataset(sub_seq, sub_mask)
        batch_size = 32
        # dataLoader for validation set
        self.sub_dataloader = DataLoader(self.sub_data,batch_size=batch_size)

# Measure how long the evaluation going to takes.

    def pred(self):
        t0 = time.time()

        for step, batch in enumerate(self.sub_dataloader):
    # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
                elapsed = (time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.sub_dataloader), elapsed))
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            with torch.no_grad():
                outputs = self.model(b_input_ids, b_input_mask)
                pred_probs = torch.sigmoid(outputs.logits)
                if step == 0:
                    predictions = pred_probs.cpu().detach().numpy()
                else:
                    predictions = np.append(predictions, pred_probs.cpu().detach().numpy(), axis=0)


        categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        predictions_df = pd.DataFrame(predictions, columns = categories)
        print(predictions_df)
