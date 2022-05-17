

def isNet(obj):
    try:
        # might work better with isInstance
        if (type(obj['places'])==set) and (type(obj['transitions'])==set): # and obj['arcs']:
            #Check that all Transitions are strings
            for elem in obj['transitions']:
                if not isinstance(elem,str):
                    return False

            #Check that Places and Transitions are disjoint aswell as Places being strings
            for elem in obj['places']:
                if not isinstance(elem,'str') | (elem in obj['transitions']):
                    return False

            #Check that inputArcs are always Place -> Transition
            for end in obj['inputArcs']:
                if end not in obj['transitions']:
                    return False

                for start in obj['inputArcs'][end]:
                    if start not in obj['places']:
                        return False

            #Check that outputArcs are always Transition -> Place
            for start in obj['outputArcs']:
                if start not in obj['transitions']:
                    return False

                for end in obj['outputArcs'][start]:
                    if end not in obj['places']:
                        return False

            #Checks that resetArcs are always Transition -> Place
            for start in obj['resetArcs']:
                if start not in obj['transitions']:
                    return False
                for end in obj['resetArcs'][start]:
                    if end not in obj['places']:
                        return False

            return True
        else:
            return False
    except:
        return False

def isPetriNet(obj):
    net = obj['net']
    if isNet(net):
        try:
            for key in obj['marking']:
                if (key not in net['places']) or (type(obj['marking'][key] != int or float)):
                    return False
            return True
        except:
            return False
    else:
        return False



