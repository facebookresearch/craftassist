# define the grammar
# cf https://docs.google.com/presentation/d/1gzC878kkIgDL015c-kLFXEBc1SAuq_XOsheQowsyIEA/edit?usp=sharing

#####
## define node types
class InternalNode(Object):

    def __init__(self, name):
        self.name                   = name
        self.node_choices           = []
        self.intern_node_children   = {}
        self.categorical_children   = {}
        seld.span_children          = {}
        self.node_value             = False


class SpanLeaf(Object):

    def __init__(self):
        self.span_value = False


class CategoricalLeaf(Object):

    def __init__(self):
        self.choices    = []
        self.cat_value  = False


#####
# define leaves
class LocationTypeLeaf(CategoricalLeaf):

    def __init__(self, absolute):
        super(LocationTypeLeaf).__init__(self)
        self.choices    += ['Coordinates', 'AgentPos',
                            'SpeakerPos', 'SpeakerLook']
        if not self.absolute:
            self.choices    += ['BlockObject', 'Mob']


class ConditionTypeLeaf(CategoricalLeaf):

    def __init__(self):
        super(ConditionTypeLeaf).__init__(self)
        self.choices    += ['AdjacentToBlockType', 'Never']


class RepeatTypeLeaf(CategoricalLeaf):

    def __init__(self):
        super(RepeatTypeLeaf).__init__(self)
        self.choices    += ['FOR', 'ALL']


class RepeatDirectionLeaf(CategoricalLeaf):

    def __init__(self):
        super(RepeatDirectionLeaf).__init__(self)
        self.choices    += ['RIGHT', 'UP']  # TODO: check with Kavya


class RelativeDirectionLeaf(CategoricalLeaf):

    def __init__(self):
        super(RelativeDirectionLeaf).__init__(self)
        self.choices    += ['LEFT' , 'RIGHT', 'UP', 'DOWN',
                            'FRONT', 'BACK', 'AWAY']


#####
# build tree
def make_repeat_node(name):
    repeat        = InternalNode(name)
    repeat.node_choices                           = [True, False]
    repeat.categorical_children['repeat_type']    = RepeatTypeLeaf()
    repeat.categorical_children['repeat_dir']     = RepeatTypeLeaf()
    repeat.span_children['repeat_count']          = SpanLeaf()
    return repeat


def make_location_node(loc_name, ref_name,
                       ref_loc_name, repeat_name):
    repeat          = make_repeat_node(repeat_name)
    lr_location     = InternalNode(ref_loc_name)
    lr_location.node_choices                            = [True, False]
    lr_location.categorical_children['location_type']   = LocationTypeLeaf(True)
    lr_location.span_children['coordinates']            = SpanLeaf()
    l_ref_object    = InternalNode(ref_name)
    l_ref_object.node_choices                           = [True, False]
    l_ref_object.intern_node_children[ref_loc_name]     = lr_location
    l_ref_object.intern_node_children[repeat_name]      = lr_repeat
    l_ref_object.span_children['has_name_']             = SpanLeaf()
    l_ref_object.span_children['has_colour_']           = SpanLeaf()
    l_ref_object.span_children['has_size_']             = SpanLeaf()
    location        = InternalNode(loc_name)
    location.node_choices                               = [True, False]
    location.intern_node_children[ref_name]             = l_ref_object
    location.categorical_children['location_type']      = LocationTypeLeaf(False)
    location.categorical_children['relative_direction'] = RelativeDirectionLeaf()
    location.span_children['coordinates']               = SpanLeaf()
    return location

    
def make_full_action()
    # ACTION_LOCATION
    action_location = make_location_node('action_location', 'al_ref_object',
                                         'alr_location', 'alr_repeat')
    # STOP_CONDITION
    stop_condition  = InternalNode('stop_condition')
    stop_condition.node_choices                             = [True, False]
    stop_condition.categorical_children['condition_type']   = ConditionTypeLeaf(True)
    stop_condition.span_children['block_type']              = SpanLeaf()
    # SCHEMATIC
    s_repeat        = make_repeat_node('s_repeat')
    schematic       = InternalNode('schematic')
    schematic.node_choices                      = [True, False]
    schematic.intern_node_children['s_repeat']  = s_repeat
    for k in ["has_block_type_" , "has_name_", "has_attribute_", "has_size_" , "has_orientation_",
              "has_thickness_", "has_colour_", "has_height_", "has_length_", "has_radius_",
              "has_slope_", "has_width_", "has_base_", "has_distance_"]:
        schematic.span_children[k]              = SpanLeaf()
    # ACTION_REPEAT
    action_repeat   = make_repeat_node('action_repeat')
    # ACTION_REF_OBJECT
    ar_location     = make_location_node('ar_location', 'arl_ref_object',
                                         'arlr_location', 'arlr_repeat')
    ar_repeat       = InternalNode('ar_repeat')
    ar_repeat.node_choices                          = [True, False]
    ar_repeat.categorical_children['repeat_type']   = RepeatTypeLeaf()
    ar_repeat.categorical_children['repeat_dir']    = RepeatTypeLeaf()
    ar_repeat.span_children['repeat_count']         = SpanLeaf()
    action_ref_object   = InternalNode('action_ref_object')
    action_ref_object.node_choices                          = [True, False]
    action_ref_object.intern_node_children['ar_location']   = ar_location
    action_ref_object.intern_node_children['ar_repeat']     = ar_repeat
    action_ref_object.span_children['has_name_']            = SpanLeaf()
    action_ref_object.span_children['has_colour_']          = SpanLeaf()
    action_ref_object.span_children['has_size_']            = SpanLeaf()
    # ROOT
    action              = InternalNode('action')
    action.node_value   = "Noop"
    action.node_choices                                 = ["Build", "Noop", "Span", "Fill",
                                                           "Destroy", "Move", "Undo", "Stop",
                                                           "Dig", "Tag", "FreeBuild", "Answer"]
    action.intern_node_children['action_location']      = action_location
    action.intern_node_children['stop_condition']       = stop_condition
    action.intern_node_children['schematic']            = schematic
    action.intern_node_children['action_repeat']        = action_repeat
    action.intern_node_children['action_ref_object']    = action_ref_object
    action.span_children['tag']                         = SpanLeaf()
    action.span_children['has_size_']                   = SpanLeaf()
    action.span_children['has_length_']                 = SpanLeaf()
    action.span_children['has_depth_']                  = SpanLeaf()
    action.span_children['has_width_']                  = SpanLeaf()
    return action

