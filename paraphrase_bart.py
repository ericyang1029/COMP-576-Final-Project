

from parrot import Parrot
import torch
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

 
# uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

def paraphrase(comment):
    #Init models (make sure you init ONLY once if you integrate this to your code)
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)
    res = []
    phrases = [ comment
    ]
    try:
        for phrase in phrases:
            para_phrases = parrot.augment(input_phrase=phrase, max_return_phrases = 4)
            for para_phrase in para_phrases:
                # print(para_phrase[0])
                res.append(para_phrase[0])
    except:
        return
    return res

# Read the CSV file
file_path = "course_eval_1.csv"
df = pd.read_csv(file_path)

# Filter rows with sentimentLabel as "neg" or "neutral"
filtered_df = df[df['sentimentLabel'].isin(['neg', 'neutral'])]

# Create a list to store new rows
new_rows = []

# Iterate through the filtered dataframe
for _, row in filtered_df.iterrows():
    original_comment = row['comment']
    paraphrased_comments = paraphrase(original_comment)
    try:
        # Add each paraphrased comment as a new row
        for paraphrased_comment in paraphrased_comments:
            new_row = row.copy()
            new_row['comment'] = paraphrased_comment
            new_row['commentLength'] = len(paraphrased_comment.split())  # Update commentLength
            # print(new_row)
            new_rows.append(new_row)
    except:
        continue

# Create a dataframe for the new rows
new_rows_df = pd.DataFrame(new_rows)

# Append the new rows to the original dataframe
updated_df = pd.concat([df, new_rows_df], ignore_index=True)

# Save the updated dataframe back to the CSV
updated_df.to_csv(file_path, index=False)

print("The file has been updated with paraphrased comments.")