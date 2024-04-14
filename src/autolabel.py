from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import yolov8
from process_image import process
def label(input_folder):
    process(input_folder)
    ontology = CaptionOntology({"person" : "person",
                                "anything with a screen or monitor that could possible be a laptop" : "laptop",
                                "any sort of place that someone could sit, this includes benches, chairs, sofa chairs etc" : "chair",
                                "table" : "table",
                                "backpack" : "backpack"})

    base_model = GroundingDINO(ontology = ontology)
    base_model.label(
      input_folder= "./dataset/images",
      output_folder="./dataset"
  )
    
label("./imgs/humandetection/1")