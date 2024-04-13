import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(4, sharey = True)
eval_df = pd.read_csv(r'tables/eval.csv')
x_axis = eval_df.columns.to_list()[1:]
eval = eval_df.iloc[9].to_list()[1:]
yolo_coco = pd.read_csv(r'tables/yolo_coco.csv')
grounding_sam = pd.read_csv(r'tables/grounding_sam.csv')
grounding_sam_with_prompting = pd.read_csv(r'tables/grounding_sam_with_prompting.csv')
ax[0].bar(x_axis, eval)
ax[0].set_title('eval')
ax[1].bar(x_axis, yolo_coco.iloc[9].to_list()[1:])
ax[1].set_title('yolo-coco')
ax[2].bar(x_axis, grounding_sam.iloc[9].to_list()[1:])
ax[2].set_title('grounding sam')
ax[3].bar(x_axis, grounding_sam_with_prompting.iloc[9].to_list()[1:])
ax[3].set_title('grounding sam with prompting')
plt.show()
plt.close()



fig, ax = plt.subplots(2, sharey = True)
eval_people_df = pd.read_csv(r'tables/eval_people.csv')
x_axis = eval_people_df.columns.to_list()[1:]
eval_people = eval_people_df.iloc[9].to_list()[1:]
yolo_visdrone = pd.read_csv(r'tables/yolo_visdrone.csv')
ax[0].bar(x_axis, eval_people)
ax[0].set_title('eval')
ax[1].bar(x_axis, yolo_visdrone.iloc[9].to_list()[1:])
ax[1].set_title('yolo-visdrone')

# plt.show()


print(yolo_coco.to_html(index=False))
print(eval_df.to_html(index=False))
print(grounding_sam.to_html(index=False))

print(yolo_visdrone.to_html(index=False))
print(eval_people_df.to_html(index=False))