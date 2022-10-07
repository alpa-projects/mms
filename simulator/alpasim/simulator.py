from alpasim.cluster import Cluster

class EventMonitor:
    """
    All the entities must implement next_event_time and handle_event property in
    order to be monitored by the EventMonitor.
    """
    def __init__(self, entities):
        for entity in entities:
            self.check_entity(entity)
        self.entities = entities
    
    def check_entity(self, entity):
        if getattr(entity, "next_event_time", None) is None:
            raise TypeError(f"{entity} does not have a next_event_time property")
        if getattr(entity, "handle_event", None) is None or not callable(entity.handle_event):
            raise TypeError(f"{entity} does not have a handle_event method")

    def add_entity(self, entity):
        self.check_entity(entity)
        self.entities.append(entity)
    
    def next_event_entity(self):
        """
        Return the entity with smallest next_event_time.
        If all the entities have infinite next_event_time, return None.
        """
        assert len(self.entities) > 0, "EventMonitor is empty"
        next_entity = self.entities[0]
        for entity in self.entities:
            if entity.next_event_time < next_entity.next_event_time:
                next_entity = entity
        if next_entity.next_event_time == float('inf'):
            return None
        return next_entity


class Simulator:
    def __init__(self, scheduler, cluster: Cluster):
        self.monitor = EventMonitor(cluster.get_all_gpus())
        self.monitor.add_entity(scheduler)
    
    def start(self):
        while True:
            entity = self.monitor.next_event_entity()
            if entity is None:
                break
            entity.handle_event()