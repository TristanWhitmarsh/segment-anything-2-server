from flask import Flask, request, jsonify
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from collections import OrderedDict
import torch.nn.functional as F
from collections import OrderedDict
import logging

app = Flask('sam2_server')

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

mask_dict = None
inference_state = None
predictor = None
combined_mask = None
video_segments = None
device = None

def init_state(
        images,
        video_height,
        video_width,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        device=torch.device("cuda")
    ):
        """Initialize a inference state."""
        global inference_state
        global predictor

        compute_device = device  # device of the model
        inference_state = {}
        
        # Assuming `images` is your input image data as a numpy array or other format
        images = images.astype(np.float32) / 255.0 # Normalize to [0, 1] range

        # Convert to tensor and move to GPU
        image_tensor = torch.tensor(images).float().permute(0, 3, 1, 2)

        # Resize the image tensor to [batch_size, channels, image_size, image_size]
        image_size = 1024
        image_tensor = F.interpolate(image_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)

        # Define the mean and standard deviation for normalization (RGB channels)
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        if not offload_video_to_cpu:
            image_tensor = image_tensor.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)

        # Normalize the image tensor
        image_tensor = (image_tensor - img_mean) / img_std

        inference_state["images"] = image_tensor
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        #self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state


def clear_object_data(inference_state, obj_id):
    """Clear all data associated with a specific object ID."""
    obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
    if obj_idx is None:
        # If the object ID is not found, just return without doing anything
        return

    # Clear point inputs and mask inputs for all frames associated with this object ID
    inference_state["point_inputs_per_obj"].pop(obj_idx, None)
    inference_state["mask_inputs_per_obj"].pop(obj_idx, None)

    # Clear the outputs related to this object ID
    inference_state["output_dict_per_obj"].pop(obj_idx, None)

    # Clear temporary outputs related to this object ID
    inference_state["temp_output_dict_per_obj"].pop(obj_idx, None)

    # Clear the mapping of object ID to index
    inference_state["obj_id_to_idx"].pop(obj_id, None)
    inference_state["obj_idx_to_id"].pop(obj_idx, None)

    # Remove the object ID from the list of tracked object IDs
    inference_state["obj_ids"].remove(obj_id)

    # Ensure that the object ID is not marked in any tracked frames
    frames_to_clear = [frame for frame, data in inference_state["frames_already_tracked"].items() if data["obj_id"] == obj_id]
    for frame in frames_to_clear:
        inference_state["frames_already_tracked"].pop(frame, None)



def remove_points_for_frame(inference_state, frame_idx, obj_id):
    """Remove all points associated with a specific object ID in a specific frame."""
    obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
    if obj_idx is None:
        raise ValueError(f"Object ID {obj_id} not found in the inference state.")

    # Remove points for the specific frame and object
    if frame_idx in inference_state["point_inputs_per_obj"].get(obj_idx, {}):
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx)

    # Remove any mask inputs associated with the specific frame and object
    if frame_idx in inference_state["mask_inputs_per_obj"].get(obj_idx, {}):
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx)

    # Optionally, remove any output data associated with the frame and object
    if frame_idx in inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]:
        inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"].pop(frame_idx)

    if frame_idx in inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]:
        inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(frame_idx)

    # Ensure the frame is no longer marked as tracked for this object
    if frame_idx in inference_state["frames_already_tracked"]:
        inference_state["frames_already_tracked"].pop(frame_idx, None)

    print(f"Removed all points and related data for object ID {obj_id} in frame {frame_idx}")


def sam2_inference(predictor, inference_state, box, points=None, labels=None, ann_frame_idx=0, ann_obj_id=1):
    global combined_mask
    global video_segments
    # Reset the state for the specific frame
    # if box is not None and box.size > 0:
    #     clear_object_data(inference_state, ann_obj_id)

    empty_box_and_points = (box is None or box.size == 0) and (points is None or points.size == 0)


        
    if not inference_state["tracking_has_started"] and not empty_box_and_points:
        clear_object_data(inference_state, ann_obj_id)

    if not inference_state["tracking_has_started"] and empty_box_and_points:
        clear_object_data(inference_state, ann_obj_id)
        if combined_mask is None:
            combined_mask = np.zeros((inference_state['video_height'], inference_state['video_width']), dtype=np.uint8)
        else:
            combined_mask[combined_mask == ann_obj_id] = 0
        return combined_mask
        
    if inference_state["tracking_has_started"] and empty_box_and_points:
        # Create a combined mask for the current frame using all IDs
        if ann_frame_idx in video_segments:
            frame_segments = video_segments[ann_frame_idx]
            combined_mask = np.zeros((inference_state['video_height'], inference_state['video_width']), dtype=np.uint8)
            for out_obj_id, out_mask in frame_segments.items():
                # Ensure that the mask is only applied to areas where the combined_mask is still 0
                combined_mask[(out_mask > 0) & (combined_mask == 0)] = out_obj_id
            return combined_mask
        else:
            # If no segments for this frame, return a zeroed mask for this frame
            return np.zeros((inference_state['video_height'], inference_state['video_width']), dtype=np.uint8)


    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
            box=box,
            clear_old_points=True,
        )
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        raise

    combined_mask = np.zeros((inference_state['video_height'], inference_state['video_width']), dtype=np.uint8)

    for mask_logits, id in zip(out_mask_logits, out_obj_ids):
        # Add a channel dimension back before interpolation
        mask_logits = mask_logits.unsqueeze(0)  # Adds a channel dimension (C=1)
        
        mask_logits = F.interpolate(mask_logits, size=(inference_state['video_height'], inference_state['video_width']), mode='bilinear', align_corners=False)
        
        # Move tensor to CPU before converting to NumPy
        mask = (mask_logits > 0.0).squeeze().cpu().numpy()  # Remove the channel dimension after interpolation

        combined_mask[(mask > 0) & (combined_mask == 0)] = id

    return combined_mask


def sam2_inference_3D(predictor, inference_state):
    global video_segments
    video_segments = {}

    # Force state consolidation
    predictor.propagate_in_video_preflight(inference_state)

    # Forward tracking (reverse=False)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state=inference_state,
        reverse=False
    ):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = F.interpolate(
                out_mask_logits[i].unsqueeze(0),  # Adds a channel dimension (C=1)
                size=(inference_state['video_height'], inference_state['video_width']),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()  # Remove the channel dimension after interpolation
            video_segments[out_frame_idx][out_obj_id] = mask

    # Reverse tracking (reverse=True)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state=inference_state,
        reverse=True
    ):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = F.interpolate(
                out_mask_logits[i].unsqueeze(0),  # Adds a channel dimension (C=1)
                size=(inference_state['video_height'], inference_state['video_width']),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()  # Remove the channel dimension after interpolation
            # Combine the reverse mask with the existing forward mask, if any
            if out_obj_id in video_segments[out_frame_idx]:
                # Combine masks - typically, you would use logical OR, or keep the latest mask
                video_segments[out_frame_idx][out_obj_id] = np.maximum(
                    video_segments[out_frame_idx][out_obj_id], mask
                )
            else:
                video_segments[out_frame_idx][out_obj_id] = mask

    # Get the number of frames and initialize the slices array
    num_frames = inference_state["num_frames"]
    slices = np.zeros((num_frames, inference_state['video_height'], inference_state['video_width']), dtype=np.int8)

    # Place the masks in the correct location in slices based on the out_frame_idx
    for out_frame_idx in sorted(video_segments.keys()):
        masks = video_segments[out_frame_idx]
        for out_obj_id, out_mask in masks.items():
            # Ensure that the mask is only applied to areas where the slices array is still 0
            slices[out_frame_idx][(out_mask > 0) & (slices[out_frame_idx] == 0)] = out_obj_id

    print(slices.shape)

    return slices


@app.route('/infer3D', methods=['POST'])
def infer3D():
    print("Running full inference")
    global inference_state
    global predictor
    masks = sam2_inference_3D(predictor, inference_state)

    # Send the mask as a JSON response
    pixel_values = masks.flatten().tolist()
    return jsonify({"width": masks.shape[1], "height": masks.shape[0], "pixels": pixel_values})


@app.route('/infer', methods=['POST'])
def infer():
    print("Running inference")
    global inference_state


    width = int(request.form['width'])
    height = int(request.form['height'])
    frames = int(request.form['frames'])

    frame = int(request.form['frame'])
    points_str = request.form.get('input_points', '')
    id_str = request.form['input_id']
    labels_str = request.form.get('input_labels', '')
    box_str = request.form.get('box', '')

    # Process the box if provided
    box = None
    if box_str != '':
        box_list = box_str.split(';')
        box = np.array([[int(x) for x in box[1:-1].split(',')] for box in box_list])

    
        # Check if points_str is not empty
    if points_str != '':
        # Split points and labels by semicolon
        points_list = points_str.split(';')
        labels_list = labels_str.split(';')

        # Initialize empty lists to accumulate points and labels
        points_accumulated = []
        labels_accumulated = []

        for point, label in zip(points_list, labels_list):
            if point.strip() and label.strip():  # Ensure point and label are not empty
                points_array = np.fromstring(point, sep=',')
                points_accumulated.append(points_array)
                labels_accumulated.append(int(label))

        # Convert accumulated lists to numpy arrays
        points = np.array(points_accumulated).astype(int)
        labels = np.array(labels_accumulated).astype(int)

    else:
        points = None
        labels = None

    # Convert the list of strings to a list of integers
    id = int(id_str)
    
    # if box is not None and box.size > 0:
    #     print(box)
    # if points is not None and points.size > 0:
    #     print(points)
    # if labels is not None and labels.size > 0:
    #     print(labels)
    # print(id)

    masks = sam2_inference(predictor, inference_state, box, points, labels, frame, id)

    # Send the mask as a JSON response
    pixel_values = masks.flatten().tolist()
    return jsonify({"width": masks.shape[1], "height": masks.shape[0], "pixels": pixel_values})


@app.route('/add_mask', methods=['POST'])
def add_mask():
    print("Adding mask")
    global inference_state
    global predictor

    try:
        # Get the width, height, and frame from the request
        width = int(request.form['width'])
        height = int(request.form['height'])
        frame = int(request.form['frame'])

        # Retrieve the mask data from the request
        if 'mask' in request.files:
            file = request.files['mask']
            mask_data = np.frombuffer(file.read(), dtype=np.uint8).reshape((height, width))

            # Debug: Print received mask data details
            print(f"Received mask data shape: {mask_data.shape}")
            print(f"Unique values in mask: {np.unique(mask_data)}")

            # Loop through each unique object ID in the mask (excluding the background 0)
            unique_ids = np.unique(mask_data)
            unique_ids = unique_ids[unique_ids != 0]  # Exclude background

            for obj_id in unique_ids:
                # Create a binary mask for the current object ID
                binary_mask = (mask_data == obj_id).astype(np.float32)

                # Debug: Print mask tensor shape before resizing
                print(f"Initial mask shape: {binary_mask.shape}")

                # Resize the mask to match the image dimensions in the inference state
                mask_tensor = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(inference_state["device"])
                mask_tensor = F.interpolate(mask_tensor, size=(inference_state['video_height'], inference_state['video_width']), mode='nearest')

                # Remove batch and channel dimensions to meet expected input shape
                mask_tensor = mask_tensor.squeeze()

                # Debug: Print mask tensor shape after resizing and squeezing
                print(f"Processed mask shape: {mask_tensor.shape}")

                # Call the predictor to add the new mask for this object ID
                predictor.add_new_mask(inference_state=inference_state, frame_idx=frame, mask=mask_tensor, obj_id=obj_id)

            # Return a success response
            return jsonify({"status": "success", "message": "Mask(s) added successfully"}), 200
        else:
            return jsonify({"status": "error", "message": "No mask data provided"}), 400

    except Exception as e:
        import traceback
        print(f"Error processing mask: {e}")
        traceback.print_exc()  # Print the full traceback for detailed error info
        return jsonify({"status": "error", "message": str(e)}), 500


    except Exception as e:
        import traceback
        print(f"Error processing mask: {e}")
        traceback.print_exc()  # Print the full traceback for detailed error info
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    print("Resetting inference state")
    global inference_state
    global predictor
    global mask_dict
    mask_dict = OrderedDict()

    try:
        predictor.reset_state(inference_state)
        return jsonify({"status": "success", "message": "Inference state reset"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/init', methods=['POST'])
def init():
    print("Initializing")

    global inference_state
    global predictor
    global device

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    width = int(request.form['width'])
    height = int(request.form['height'])
    frames = int(request.form['frames'])
    model = request.form['model']

    checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    if model == "sam2_hiera_tiny":
        checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"

    if model == "sam2_hiera_small":
        checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"

    if model == "sam2_hiera_base_plus":
        checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b.yaml"

    if model == "sam2_hiera_large":
        checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

    print("Loading model " + model)

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    print("Loading model done")
    try:
        # Reconstruct the image from raw RGB pixel data
        if 'pixels' in request.files:
            file = request.files['pixels']
            image_data = np.frombuffer(file.read(), dtype=np.uint8).reshape((frames, height, width, 3))
            images = [Image.fromarray(image_data[i], 'RGB') for i in range(frames)]
            images = np.array(images)

            inference_state = init_state(images, height, width, device=device)
            predictor.reset_state(inference_state)

            print("Ready")

        # Return a success message with some information
        return jsonify({"status": "success", "message": "Inference state initialized", "frames": frames}), 200

    except Exception as e:
        # Handle any errors and return a failure response
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/infer2D', methods=['POST'])
def infer2D():
    print("Running 2D inference")

    global predictor
    global mask_dict

    points_str = request.form.get('input_points', '')
    id_str = request.form['input_id']
    labels_str = request.form.get('input_labels', '')
    box_str = request.form.get('box', '')

    # Process the box if provided
    box = None
    if box_str != '':
        box_list = box_str.split(';')
        box = np.array([[int(x) for x in box[1:-1].split(',')] for box in box_list])

    
        # Check if points_str is not empty
    if points_str != '':
        # Split points and labels by semicolon
        points_list = points_str.split(';')
        labels_list = labels_str.split(';')

        # Initialize empty lists to accumulate points and labels
        points_accumulated = []
        labels_accumulated = []

        for point, label in zip(points_list, labels_list):
            if point.strip() and label.strip():  # Ensure point and label are not empty
                points_array = np.fromstring(point, sep=',')
                points_accumulated.append(points_array)
                labels_accumulated.append(int(label))

        # Convert accumulated lists to numpy arrays
        points = np.array(points_accumulated).astype(int)
        labels = np.array(labels_accumulated).astype(int)

    else:
        points = None
        labels = None

    # Convert the list of strings to a list of integers
    id = int(id_str)

    masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=False,
        )
    
    mask = masks[0,...].astype(np.int8)
    
    mask_dict[id] = mask
    mask_dict.move_to_end(id)

    # Assuming all masks have the same shape
    combined_mask = np.zeros_like(mask).astype(np.int8)

    for id, mask in mask_dict.items():
        mask = mask.astype(np.int8)
        combined_mask = np.where(combined_mask == 0, mask * id, combined_mask)


    # Send the mask as a JSON response
    pixel_values = combined_mask.flatten().tolist()
    return jsonify({"width": combined_mask.shape[1], "height": combined_mask.shape[0], "pixels": pixel_values})


@app.route('/init2D', methods=['POST'])
def init2D():
    print("Initializing for 2D")

    global predictor
    global mask_dict
    mask_dict = OrderedDict()

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    width = int(request.form['width'])
    height = int(request.form['height'])
    model = request.form['model']

    checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    if model == "sam2_hiera_tiny":
        checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"

    if model == "sam2_hiera_small":
        checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"

    if model == "sam2_hiera_base_plus":
        checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b.yaml"

    if model == "sam2_hiera_large":
        checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

    print("Loading model " + model)

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    predictor.model.to(device)
    predictor.model.eval()

    image = None

    print("Loading model done")
    try:
        if 'pixels' in request.files:
            file = request.files['pixels']
            image_data = np.frombuffer(file.read(), dtype=np.uint8).reshape((height, width, 3))
            image = Image.fromarray(image_data, 'RGB')

            predictor.set_image(image)

            print("Ready")

        # Return a success message with some information
        return jsonify({"status": "success", "message": "Inference state initialized"}), 200

    except Exception as e:
        # Handle any errors and return a failure response
        return jsonify({"status": "error", "message": str(e)}), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
