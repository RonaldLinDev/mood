from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import yolov8
from process_data import dataset_sublabel
def label(correct_label, input_dir, flipped = False):
    data = dataset_sublabel(correct_label, input_dir = input_dir, flipped = flipped)
    print('converting')
    data.convert_to_jpg()
    ontology = CaptionOntology({"person" : "person",
                                "anything with a screen or monitor that could possible be a laptop" : "laptop",
                                "any sort of place that someone could sit, this includes benches, chairs, sofa chairs etc" : "chair",
                                "table" : "table",
                                "backpack" : "backpack"})

    base_model = GroundingDINO(ontology = ontology)
    base_model.label(
      input_folder= "./dataset/images",
      output_folder= "./dataset"
    )
    data.verify()
    
# label('person', "./imgs/humandetection/1/")
# label('person', "./imgs/humandetection/0/", flipped = True)
# label('chair', 'imgs/office/Chair/')
# label('laptop', 'imgs/office/Laptop/')
# label('table', 'imgs/office/Table/')

# label('backpack', './imgs/domain/back_pack/')
# label('chair', './imgs/domain/desk_chair/')
# label('laptop', './imgs/domain/laptop_computer/')
# label('laptop', './imgs/domain/laptop_computer/')


label('table', './imgs/indoor/table/')

label('person', './imgs/v47/1.01/')
label('person', './imgs/v47/2.01/')
label('person', './imgs/v47/3.01/')
label('person', './imgs/v47/4.01/')