from pm4py.objects.dcr.extended.obj import ExtendedDcrGraph
from typing import Set, Dict


class HierarchicalDcrGraph(ExtendedDcrGraph):
    """
    Extends the MilestoneNoResponseDcrGraph to include nested groups
    and subprocesses, supporting hierarchical structures in DCR Graphs.

    Attributes
    ----------
    self.__nestedgroups: Dict[str, Set[str]]
        Maps group names to sets of event IDs.
    self.__subprocesses: Dict[str, Set[str]]
        Maps subprocess names to sets of event IDs.
    self.__nestedgroups_map: Dict[str, str]
        Maps event IDs to their group names.
    self.__subprocess_map: Dict[str, str]
        Maps event IDs to their subprocess names.

    Methods
    -------
    obj_to_template(self) -> dict:
        Returns a dictionary representation of the graph, including hierarchy info.
    """

    def __init__(self, template=None):
        super().__init__(template)
        self.__nestedgroups = {} if template is None else template.get('nestedgroups', {})
        self.__subprocesses = {} if template is None else template.get('subprocesses', {})
        self.__nestedgroups_map = {} if template is None else template.get('nestedgroupsMap', {})
        self.__subprocess_map = {} if template is None else template.get('subprocessMap', {})

        # Build nestedgroups_map if missing
        if not self.__nestedgroups_map and self.__nestedgroups:
            self.__nestedgroups_map = {
                e: group for group, events in self.__nestedgroups.items() for e in events
            }

        # Build subprocess_map if missing
        if not self.__subprocess_map and self.__subprocesses:
            self.__subprocess_map = {
                e: sp for sp, events in self.__subprocesses.items() for e in events
            }

    def obj_to_template(self):
        res = super().obj_to_template()
        res['nestedgroups'] = self.__nestedgroups
        res['subprocesses'] = self.__subprocesses
        res['nestedgroupsMap'] = self.__nestedgroups_map
        res['subprocessMap'] = self.__subprocess_map
        return res

    @property
    def nestedgroups(self) -> Dict[str, Set[str]]:
        return self.__nestedgroups

    @nestedgroups.setter
    def nestedgroups(self, ng):
        self.__nestedgroups = ng

    @property
    def nestedgroups_map(self) -> Dict[str, str]:
        return self.__nestedgroups_map

    @nestedgroups_map.setter
    def nestedgroups_map(self, ngm):
        self.__nestedgroups_map = ngm

    @property
    def subprocesses(self) -> Dict[str, Set[str]]:
        return self.__subprocesses

    @subprocesses.setter
    def subprocesses(self, sps):
        self.__subprocesses = sps

    @property
    def subprocess_map(self) -> Dict[str, str]:
        return self.__subprocess_map

    @subprocess_map.setter
    def subprocess_map(self, spm):
        self.__subprocess_map = spm

    @property
    def hierarchy_map(self) -> Dict[str, Dict[str, str]]:
        """
        Returns a merged view of nestedgroups_map and subprocess_map,
        assuming an event can belong to only one: either a group or a subprocess.

        Format:
        {
            "event_id": {"type": "group" | "subprocess", "name": "group_or_subprocess_name"},
            ...
        }
        """
        merged = {}

        # Add group entries
        for event_id, group in self.__nestedgroups_map.items():
            merged[event_id] = {"type": "group", "name": group}

        # Add subprocess entries (only if not already in groups)
        for event_id, subprocess in self.__subprocess_map.items():
            if event_id in merged:
                raise ValueError(
                    f"Conflict: Event '{event_id}' cannot belong to both a group and a subprocess."
                )
            merged[event_id] = {"type": "subprocess", "name": subprocess}

        return merged

