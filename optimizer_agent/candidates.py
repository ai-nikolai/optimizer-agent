import textdistance


class SingleCandidate():
    def __init__(self, path, scores, other=None, prompt_data=None, **kwargs):
        self.path=path
        self.score = scores["combined_score"]
        self.scores=scores
        self.other=other
        self.prompt_data=prompt_data
        # self.response_data=response_data
        self.kwargs=kwargs


class Candidates():
    def __init__(self,**kwargs):
        self.candidates = []
        self.best = {}
        self.name="DEFAULT CLASS"
    
    def get_best(self):
        raise NotImplementedError()
    
    def update_candidates(self, candidate, *argv, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        representation = "-- Candidates Object: --\n"
        representation += f"Name: {self.name}\n{{\n"
        best_candidates = self.get_best()
        for key,value in best_candidates.items():
            representation += f'"{key}" : "{value.score}",\n'
        
        representation += "}\n--end--"
        return representation

class CandidatesBestNAll(Candidates):
    def __init__(self,**kwargs):
        self.candidates = []
        self.best = {}
        self.name="bestN_all"
    
    def get_best(self):
        return self.best
    
    def update_candidates(self, candidate, *argv, **kwargs):
        self.candidates.append(candidate)

        if self.best:
            if self.best[self.name].score < candidate.score:
                self.best[self.name] = candidate
        else:
            self.best[self.name] = candidate

class CandidatesBestNLast(Candidates):
    def __init__(self,**kwargs):
        self.candidates = []
        self.best = {}
        self.current_iteration = None
        self.name="bestN_last"

    def get_best(self):
        return self.best
    
    def update_candidates(self, candidate, current_iteration, *argv, **kwargs):
        if not self.current_iteration:
            self.candidates = [candidate]
            self.current_iteration = current_iteration
        else:
            if current_iteration > self.current_iteration:
                self.candidates = [candidate]
                self.current_iteration = current_iteration
            else:
                self.candidates.append(candidate)
        
        if self.best:
            if self.best[self.name].score < candidate.score:
                self.best[self.name] = candidate
        else:
            self.best[self.name] = candidate


class MapElitesCandidates(Candidates):
    def __init__(self, num_map_elites, **kwargs):
        self.candidates = {}
        self.best = {}
        self.best_all = {}
        self.best_scores = []
        self.num_map_elites = num_map_elites
        self.name = "map_elites"

        self.cluster_threshold = 0.2

    
    def get_best(self): 
        """Finds the top map elites"""
        self.best = {}
        self.best_scores = sorted(self.best_scores, key=lambda x:x[1])
        self.best_scores.reverse()

        num_actual_elites = min(self.num_map_elites, len(self.best_scores))
        for jdx in range(num_actual_elites):
            best_key = self.best_scores[jdx][0]
            self.best[best_key] = self.best_all[best_key]
        
        return self.best

    def _insert_into_best_scores(self,existing_key, new_score):
        for idx, (key,val) in enumerate(self.best_scores):
            if key==existing_key:
                self.best_scores.pop(idx)
                break
        self.best_scores.append((existing_key,new_score))

    def update_candidates(self, candidate, *argv, **kwargs):
        """Updates map elites... based on 'other' key """
        new_key = candidate.other

        # this map elites already exists
        if new_key in self.candidates:
            self.candidates[new_key].append(candidate)
            if self.best_all[new_key].score < candidate.score:
                self.best_all[new_key] = candidate
                self._insert_into_best_scores(new_key,candidate.score)
        
        # this is a new cluster
        else:
            self.candidates[new_key]=[candidate]
            self.best_all[new_key] = candidate
            self.best_scores.append((new_key, candidate.score))
    
    def get_key(self, new_generation, current_other):
        """Returns the key to use for a candidate..."""
        
        if new_generation == current_other:
            return current_other, -0.0
        
        elif not self.best_all:
            return current_other, -1.0
        
        else:
            # Distance based only for now...
            return_other = current_other
            # https://pypi.org/project/textdistance/
            minimum = textdistance.jaccard.normalized_distance(new_generation, current_other)
            for key,_ in  self.best_all.items():
                distance = textdistance.jaccard.normalized_distance(key, new_generation)
                if distance < minimum:
                    return_other = key
                    minimum = distance
            
            if minimum > self.cluster_threshold:
                return new_generation, minimum
            else:
                return return_other, minimum


        # TODO:
        # 1. add strategy overwrite for map-elites ("ideas")
        # 2. add diversity metric
        # 3. add static classification for map-elites
        # 4. add dynamic classification for map-elites
        # 5. probably move to map elites class...
        # do something...
