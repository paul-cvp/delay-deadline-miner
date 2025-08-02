import pandas as pd
from datetime import timedelta
from collections import defaultdict
from copy import deepcopy

def abstract_events(log_df: pd.DataFrame, abstraction_map: dict) -> pd.DataFrame:
    # Reverse the abstraction_map to map each concrete event to its abstract event
    concrete_to_abstract = {}
    for abstract_event, concrete_events in abstraction_map.items():
        for concrete_event in concrete_events:
            concrete_to_abstract[concrete_event] = abstract_event

    # Create a copy of the dataframe to avoid modifying the original
    abstracted_df = log_df.copy()

    # Map the 'concept:name' column using the reverse dictionary
    # If an event is not found in the map, keep it as is
    abstracted_df['concept:name'] = abstracted_df['concept:name'].map(concrete_to_abstract).fillna(abstracted_df['concept:name'])

    return abstracted_df

from copy import deepcopy
from datetime import timedelta

def handle_timed_condition_subprocess(graph, delay_cutoff: timedelta = timedelta(hours=1)):
    graph = deepcopy(graph)

    # Step 1: Detect self-excludes
    self_excludes = {source for source, targets in graph.excludes.items() if source in targets}

    # Step 2: Adjust for nested groups that self-exclude
    for group, events in graph.nestedgroups.items():
        if group in self_excludes:
            self_excludes.add(group)
            self_excludes -= events  # remove children from self_excludes

    # Step 3: Remove short-timed conditions and remap problematic sources
    source_condition_remapping = {}
    for target, sources_dict in graph.timedconditions.items():
        for source, timing in list(sources_dict.items()):
            if timing < delay_cutoff:
                del graph.timedconditions[target][source]
            elif source in self_excludes:
                if source not in graph.subprocesses:
                    condition_subprocess = f'{source}_completed'
                    graph.subprocesses[condition_subprocess] = {source}
                    graph.subprocess_map[source] = condition_subprocess
                    source_condition_remapping[source] = condition_subprocess
                    graph.events.add(condition_subprocess)
                    graph.labels.add(condition_subprocess)
                    graph.label_map[condition_subprocess] = condition_subprocess
                    graph.marking.included.add(condition_subprocess)

                    for group, events in graph.nestedgroups.items():
                        if source in events:
                            events.remove(source)
                            events.add(condition_subprocess)

    # Step 4: Clean up empty timedconditions
    for k, v in list(graph.timedconditions.items()):
        if len(v) == 0:
            del graph.timedconditions[k]

    # Step 5: Replace original sources with subprocessed version in conditions
    for target, sources in graph.conditions.items():
        for source in list(sources):
            if source in source_condition_remapping:
                sources.remove(source)
                remapped = source_condition_remapping[source]
                sources.add(remapped)
                delay = graph.timedconditions[target][source]
                del graph.timedconditions[target][source]
                graph.timedconditions[target][remapped] = delay

    # Step 6: Build event-to-subprocess map
    event_to_subprocess = {}
    for sp_name, sp_events in graph.subprocesses.items():
        for e in sp_events:
            event_to_subprocess[e] = sp_name

    # Step 7: Redirect excludes (only if not self-excludes)
    for source in list(graph.excludes.keys()):
        updated_targets = set()
        for target in list(graph.excludes[source]):
            if source != target and target in event_to_subprocess:
                updated_targets.add(event_to_subprocess[target])
            else:
                updated_targets.add(target)
        graph.excludes[source] = updated_targets

    # Step 8: Redirect includes to subprocess targets
    for source in list(graph.includes.keys()):
        updated_targets = set()
        for target in list(graph.includes[source]):
            if target in event_to_subprocess:
                updated_targets.add(event_to_subprocess[target])
            else:
                updated_targets.add(target)
        graph.includes[source] = updated_targets

    # Step 9: Deduplicate timedconditions: keep only subprocess/nestedgroup timedcondition when the delay and target are the same

    hierarchical_events = set(graph.subprocesses.keys()).union(graph.nestedgroups.keys())
    atomic_events = graph.events.difference(hierarchical_events)

    for ae in atomic_events:
        for target, sources in list(graph.timedconditions.items()):
            if ae in sources.keys():
                hie = ae
                run = True
                while run:
                    if hie:
                        hie = graph.hierarchy_map.get(hie,{'name':None})['name']
                        if hie in sources and sources[hie]==sources[ae]:
                            # print(target, ae, hie)
                            del graph.timedconditions[target][ae]
                            graph.conditions[target].remove(ae)
                    else:
                        run = False
    return graph

# def handle_timed_condition_subprocess(graph, delay_cutoff: timedelta = timedelta(hours=1)):
#     graph = deepcopy(graph)
#     self_excludes = set()
#     for source,targets in graph.excludes.items():
#         if source in targets:
#             self_excludes.add(source)
#
#     for group, events in graph.nestedgroups.items():
#         if group in self_excludes:
#             self_excludes.add(group)
#             self_excludes = self_excludes.difference(events)
#
#     source_condition_remapping = {}
#     for target, sources_dict in graph.timedconditions.items():
#         for source, timing in list(sources_dict.items()):
#             if timing < delay_cutoff:
#                 del graph.timedconditions[target][source]
#             elif source in self_excludes:
#                 if source not in graph.subprocesses:
#                     condition_subprocess = f'{source}_completed'
#                     graph.subprocesses[condition_subprocess] = {source}
#                     source_condition_remapping[source] = condition_subprocess
#                     graph.events.add(condition_subprocess)
#                     graph.labels.add(condition_subprocess)
#                     graph.label_map[condition_subprocess] = condition_subprocess
#                     graph.marking.included.add(condition_subprocess)
#
#                     for group, events in graph.nestedgroups.items():
#                         if source in events:
#                             graph.nestedgroups[group].remove(source)
#                             graph.nestedgroups[group].add(condition_subprocess)
#
#     for k,v in list(graph.timedconditions.items()):
#         if len(v)==0:
#             del graph.timedconditions[k]
#
#     for target, sources in graph.conditions.items():
#         for source in sources:
#             if source in source_condition_remapping:
#                 graph.conditions[target].remove(source)
#                 graph.conditions[target].add(source_condition_remapping[source])
#                 the_delay = graph.timedconditions[target][source]
#                 del graph.timedconditions[target][source]
#                 graph.timedconditions[target][source_condition_remapping[source]] = the_delay
#     return graph

def merge_dcr_graphs(graph1, graph2):
    # Start with a deepcopy of graph1 to preserve original data
    merged_graph = deepcopy(graph1)

    # Merge events (sets)
    merged_graph.events = graph1.events.union(graph2.events)

    # Merge markings (Marking objects have sets: executed, included, pending)
    merged_graph.marking.executed = graph1.marking.executed.union(graph2.marking.executed)
    merged_graph.marking.included = graph1.marking.included.union(graph2.marking.included)
    merged_graph.marking.pending = graph1.marking.pending.union(graph2.marking.pending)

    # Helper to merge dict[str, set[str]]
    def merge_dict_of_sets(d1, d2):
        merged = defaultdict(set)
        for k, v in (d1 or {}).items():
            merged[k].update(v)
        for k, v in (d2 or {}).items():
            merged[k].update(v)
        return dict(merged)

    # Merge relations using properties
    merged_graph.conditions = merge_dict_of_sets(graph1.conditions, graph2.conditions)
    merged_graph.responses = merge_dict_of_sets(graph1.responses, graph2.responses)
    merged_graph.includes = merge_dict_of_sets(graph1.includes, graph2.includes)
    merged_graph.excludes = merge_dict_of_sets(graph1.excludes, graph2.excludes)
    merged_graph.noresponses = merge_dict_of_sets(graph1.noresponses, graph2.noresponses)
    merged_graph.milestones = merge_dict_of_sets(graph1.milestones, graph2.milestones)

    # Merge label_map (dict[str, str]) - newer keys overwrite older ones if conflict
    merged_label_map = graph1.label_map.copy() if graph1.label_map else {}
    merged_label_map.update(graph2.label_map or {})
    merged_graph.label_map = merged_label_map

    # Merge labels (set)
    merged_graph.labels = (graph1.labels or set()).union(graph2.labels or set())

    # Merge nested dict[str, dict[str, timedelta]] for timedconditions, timedresponses
    def merge_nested_dict(d1, d2):
        merged = defaultdict(dict)
        for k, v in (d1 or {}).items():
            merged[k].update(v)
        for k, v in (d2 or {}).items():
            merged[k].update(v)
        return dict(merged)

    merged_graph.timedconditions = merge_nested_dict(graph1.timedconditions, graph2.timedconditions)
    merged_graph.timedresponses = merge_nested_dict(graph1.timedresponses, graph2.timedresponses)

    return merged_graph

def rename_events_in_dcr_graph(graph, rename_map):
    """
    Renames event strings in a DCR graph object according to rename_map.

    Parameters
    ----------
    graph : DcrGraph or subclass
        The DCR graph object to modify.
    rename_map : dict
        Dictionary mapping old event names to new event names.

    Returns
    -------
    graph : DcrGraph or subclass
        The modified graph with renamed events.
    """
    # Rename events set
    graph.events = set(rename_map.get(e, e) for e in graph.events)

    # Rename labels set
    if hasattr(graph, 'labels'):
        graph.labels = set(rename_map.get(e, e) for e in graph.labels)

    # Rename label_map
    if hasattr(graph, 'label_map'):
        graph.label_map = {rename_map.get(k, k): rename_map.get(v, v) for k, v in graph.label_map.items()}

    # Rename relations (dict[str, set[str]])
    for attr in ['conditions', 'responses', 'includes', 'excludes', 'noresponses', 'milestones']:
        if hasattr(graph, attr):
            rel = getattr(graph, attr)
            new_rel = {}
            for k, v in rel.items():
                new_k = rename_map.get(k, k)
                new_v = set(rename_map.get(e, e) for e in v)
                new_rel[new_k] = new_v
            setattr(graph, attr, new_rel)

    # Rename timed relations (dict[str, dict[str, timedelta]])
    for attr in ['timedconditions', 'timedresponses']:
        if hasattr(graph, attr):
            rel = getattr(graph, attr)
            new_rel = {}
            for k, v in rel.items():
                new_k = rename_map.get(k, k)
                new_v = {rename_map.get(e, e): t for e, t in v.items()}
                new_rel[new_k] = new_v
            setattr(graph, attr, new_rel)

    # Rename marking sets
    if hasattr(graph, 'marking'):
        for m_attr in ['executed', 'included', 'pending']:
            if hasattr(graph.marking, m_attr):
                s = getattr(graph.marking, m_attr)
                setattr(graph.marking, m_attr, set(rename_map.get(e, e) for e in s))

    # Rename subprocesses and nestedgroups
    for attr in ['subprocesses', 'nestedgroups']:
        if hasattr(graph, attr):
            rel = getattr(graph, attr)
            new_rel = {}
            for k, v in rel.items():
                new_k = rename_map.get(k, k)
                new_v = set(rename_map.get(e, e) for e in v)
                new_rel[new_k] = new_v
            setattr(graph, attr, new_rel)

    # Rename map attribute (dict[str, str])
    if hasattr(graph, 'nestedgroups_map'):
        graph.nestedgroups_map = {rename_map.get(k, k): rename_map.get(v, v) for k, v in graph.nestedgroups_map.items()}

    return graph