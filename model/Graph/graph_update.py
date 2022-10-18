import numpy as np


def is_close(embed_a, embed_b, return_prob=False, th=0.75):
    logits = np.matmul(embed_a, embed_b.transpose(1, 0))
    close = (logits > th)
    if return_prob:
        return close, logits
    else:
        return close


def update_object_graph(imggraph, objgraph, object_embedding, obs, done):
    object_score, object_category, object_mask, object_position, object_bboxes, time = \
        obs['object_score'], obs['object_category'], obs['object_mask'], obs['object_pose'], obs['object'], obs['step']
    # The position is only used for visualizations. Remove if object features are similar
    # object masking
    object_score = object_score[object_mask == 1]
    object_category = object_category[object_mask == 1]
    object_position = object_position[object_mask == 1]
    object_embedding = object_embedding[object_mask == 1]
    object_mask = object_mask[object_mask == 1]
    if done:
        objgraph.reset()
        objgraph.initialize_graph(object_embedding, object_score, object_category, object_mask, object_position)

    # not_found = not done  # Dense
    to_add = [True] * int(sum(object_mask))

    if objgraph.sparse:
        not_found = ~imggraph.found  # Sparse
    else:
        not_found = not done  # Dense
    if not_found:
        hop1_vis_node = imggraph.A[imggraph.last_localized_node_idx]
        hop1_obj_node_mask = np.sum(objgraph.A_OV.transpose(1, 0)[hop1_vis_node], 0) > 0
        curr_obj_node_mask = objgraph.A_OV[:, imggraph.last_localized_node_idx]
        neighbor_obj_node_mask = (hop1_obj_node_mask + curr_obj_node_mask) > 0
        neighbor_node_embedding = objgraph.graph_memory[neighbor_obj_node_mask]
        neighbor_obj_memory_idx = np.where(neighbor_obj_node_mask)[0]
        neighbor_obj_memory_score = objgraph.graph_score[neighbor_obj_memory_idx]
        neighbor_obj_memory_cat = objgraph.graph_category[neighbor_obj_memory_idx]

        close, prob = is_close(neighbor_node_embedding, object_embedding, return_prob=True, th=objgraph.node_th)
        for c_i in range(prob.shape[1]):
            close_mem_indices = np.where(close[:, c_i] == 1)[0]
            for m_i in close_mem_indices:
                is_same = False
                to_update = False
                if (object_category[c_i] == neighbor_obj_memory_cat[m_i]) and object_category[c_i] != -1:
                    is_same = True
                    if object_score[c_i] > neighbor_obj_memory_score[m_i]:
                        to_update = True

                if is_same:
                    to_add[c_i] = False

                if to_update:
                    objgraph.update_node(m_i, time, object_score[c_i], object_category[c_i], int(imggraph.last_localized_node_idx), object_embedding[c_i])
                    break

        # Add new objects to graph
        if sum(to_add) > 0:
            start_node_idx = objgraph.num_node()
            new_idx = np.where(np.stack(to_add))[0]
            objgraph.add_node(start_node_idx, object_embedding[new_idx], object_score[new_idx],
                                   object_category[new_idx], object_mask[new_idx], time,
                                   object_position[new_idx], int(imggraph.last_localized_node_idx))
    return objgraph


def update_image_graph(imggraph, objgraph, new_embedding, curr_obj_embeding, obs, done):
    # The position is only used for visualizations.
    position, rotation, time = obs['position'], obs['rotation'], obs['step']
    if done:
        imggraph.reset()
        imggraph.initialize_graph(new_embedding, position, rotation)

    obj_close = True
    obj_graph_mask = objgraph.graph_score[objgraph.A_OV[:, imggraph.last_localized_node_idx]] > 0.5
    if len(obj_graph_mask) > 0:
        curr_obj_mask = obs['object_score'] > 0.5
        if np.sum(curr_obj_mask) / len(curr_obj_mask) >= 0.5:
            close_obj, prob_obj = is_close(objgraph.graph_memory[objgraph.A_OV[:, imggraph.last_localized_node_idx]], curr_obj_embeding, return_prob=True, th=objgraph.node_th)
            close_obj = close_obj[obj_graph_mask, :][:, curr_obj_mask]
            category_mask = objgraph.graph_category[objgraph.A_OV[:, imggraph.last_localized_node_idx]][obj_graph_mask][:, None] == obs['object_category'][curr_obj_mask]
            close_obj[~category_mask] = False
            if len(close_obj) >= 3:
                clos_obj_p = close_obj.any(1).sum() / (close_obj.shape[0])
                if clos_obj_p < 0.1:  # Fail to localize (find the same object) with the last localized frame
                    obj_close = False

    close, prob = is_close(imggraph.last_localized_node_embedding[None], new_embedding[None], return_prob=True, th=imggraph.node_th)
    found = (np.array(done) + close.squeeze()) & np.array(obj_close).squeeze()
    found = found
    to_add = False
    if found:
        imggraph.update_node(imggraph.last_localized_node_idx, time)
        imggraph.found = True
    else:
        imggraph.found = False
        check_list = 1 - imggraph.graph_mask[:imggraph.num_node()]
        check_list[imggraph.last_localized_node_idx] = 1.0
        while not found:
            not_checked_yet = np.where((1 - check_list))[0]
            neighbor_embedding = imggraph.graph_memory[not_checked_yet]
            num_to_check = len(not_checked_yet)
            if num_to_check == 0:
                to_add = True
                break
            else:
                close, prob = is_close(new_embedding[None], neighbor_embedding, return_prob=True, th=imggraph.node_th)
                close = close[0]
                prob = prob[0]
                close_idx = np.where(close)[0]
                if len(close_idx) >= 1:
                    found_node = not_checked_yet[prob.argmax()]
                else:
                    found_node = None
                if found_node is not None:
                    found = True
                    imggraph.update_node(found_node, time, new_embedding)
                    imggraph.add_edge(found_node, imggraph.last_localized_node_idx)
                    imggraph.record_localized_state(found_node, new_embedding)
                check_list[found_node] = 1.0

    if to_add:
        new_node_idx = imggraph.num_node()
        imggraph.add_node(new_node_idx, new_embedding, time, position, rotation)
        imggraph.add_edge(new_node_idx, imggraph.last_localized_node_idx)
        imggraph.record_localized_state(new_node_idx, new_embedding)

    return imggraph