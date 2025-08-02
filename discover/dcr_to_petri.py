from copy import deepcopy

import petri_net_util as pnet_util

class Dcr2Petri:

    def __init__(self):
        self.retNet = {
            'net' : {
                'places': set(),
                'transitions': set(),
                'inputArcs': {},
                'outputArcs': {},
                'resetArcs': {},
        },
            'marking': {}
        }

    def getIncludePlace(self,event):
        return f'P_INCL_{event}'

    def getPendingPlace(self,event):
        return f'P_PEND_{event}'

    def copyMarking(self,model):
        return deepcopy(model['marking'])

    def addPlace(self,place,number:int):
        self.retNet['net']['places'].add(place)
        self.retNet['marking'][place] = number

    def dcr2Petri(self,graph):

        for event in graph['events']:
            # Initialize arc sets
            self.retNet['net']['inputArcs'][event] = set()
            self.retNet['net']['outputArcs'][event] = set()
            self.retNet['net']['resetArcs'][event] = set()
            # Add transition and two places for each event
            self.retNet['net']['transitions'].add(event)
            placeIn = self.getPendingPlace(event)
            placePe = self.getPendingPlace(event)
            self.addPlace(placeIn,1)
            self.addPlace(placePe,0)
            # placeIn needs a token for event to fire, but event doesn't consume this token
            self.retNet['net']['inputArcs'][event].add(placeIn)
            self.retNet['net']['outputArcs'][event].add(placeIn)
            # Each event resets its own pending place
            self.retNet['net']['resetArcs'][event].add(placePe)

        for startEvent in graph['includesTo']:
            for endEvent in graph['includesTo'][startEvent]:
                place = self.getIncludePlace(endEvent)
                # For startEvent ->% endEvent, startEvent resets endEvents include place
                self.retNet['net']['outputArcs'][startEvent].add(place)

        for startEvent in graph['excludesTo']:
            for endEvent in graph['excludesTo'][startEvent]:
                place = self.getIncludePlace(endEvent)
                # For startEvent -> % endEvent, startEvent resets endEvents include place
                self.retNet['net']['resetArcs'][startEvent].add(place)

        for endEvent in graph['conditionsFor']:
            for startEvent in graph['conditionsFor'][endEvent]:
                place = f'P_{startEvent}_COND_{endEvent}'
                self.addPlace(place, 0)
                # startEvent adds a token to place, so endEvent can fire
                self.retNet['net']['outputArcs'][startEvent].add(place)
                # endEvent also adds a token so it can keep firing
                self.retNet['net']['outputArcs'][endEvent].add(place)
                # endEvent needs a token in place to fire the first time
                self.retNet['net']['inputArcs'][endEvent].add(place)


        # For startEvent *-> endEvent, startEvent adds a token to the pending - place of endEvent
        for startEvent in graph['responseTo']:
            for endEvent in graph['responseTo'][startEvent]:
                place = self.getPendingPlace(endEvent)
                self.retNet['net']['outputArcs'][startEvent].add(place)

        if not pnet_util.isPetriNet(self.retNet):
            raise ValueError("Petri Net constraints broken!");

