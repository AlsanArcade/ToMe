import torch
from tome.merge import (
    bipartite_soft_matching,
    merge_source,
    merge_wavg,
    get_current_cls_token_pos_from_source,
    get_expanded_tokens_and_mask,
    transform_post_flattened_tokens_to_position_pre_flatten,
)

cls_position = torch.tensor([1, 1])

hidden_states = torch.randint(-10, 8, (2, 12, 2))

# Ziel: a,b per batch jeweils so besetzen, dass cls_token pos eine andere sein muss
# Wie:
# cls pos so, dass in b-> dann ist mgl, dass "links" token weg gemerged werden nach a,
# und beim anderen die token "rechts", sodass unterschiedliche positionen resultieren
# a gets merged into b -> b stays same size, a gets smaller
# -> put cls in b, set elems of a in one batch left of cls to merge, in the other set elems with higher indices

# a = [11,12,13,14,15,16]
# b = [21,22,23,CLS,25,26]
# ->[11,21,..,14,CLS,15..,16,26]

# now for one batch: let 11,12 merge
# by setting those to same value as 25,26
print(f"hidden_states\n{hidden_states}")
hidden_states[0, [0, 2, -1, -3], :] = 10

# in the other batch: let 15,16 merge
# by setting those to same value as 25,26
hidden_states[1, [8, 10, -1, -3], :] = 10
print(f"hidden_states new \n{hidden_states}")

hidden_states_float = hidden_states.float()

merge, _ = bipartite_soft_matching(
    metric=hidden_states_float,
    r=2,
    source=None,
    original_cls_token_positions=cls_position,
)

x = merge(hidden_states_float)
source = merge_source(merge, hidden_states_float, None)
new_pos_of_orig = source.argmax(1)
orig_pos_of_new = source.argmax(2)
print(f"new_pos_of_orig`n{new_pos_of_orig}")
print(f"orig_pos_of_new`n{orig_pos_of_new}")

new_cls_pos = get_current_cls_token_pos_from_source(cls_position, source)
c = source.argmax(1)
