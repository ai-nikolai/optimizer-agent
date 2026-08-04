class SingleCandidate():
    def __init__(self, path, score, other=None):
        self.path=path
        self.score=score
        self.other=other


class Candidates():
    def __init__(self,**kwargs):
        self.candidates = []
        self.best = None
    
    def get_best(self):
        raise NotImplementedError()
    
    def update_candidates(self, candidate, **kwargs):
        raise NotImplementedError()

class CandidatesBestNAll(Candidates):
    def __init__(self,**kwargs):
        self.candidates = []
        self.best = None
    
    def get_best(self):
        return self.best
    
    def update_candidates(self, candidate, **kwargs):
        self.candidates.append(candidate)
        if self.best:
            if best.score < candidate.score:
                self.best["bestN_all"] = candidate

class CandidatesBestNLast(Candidates):
    def __init__(self,**kwargs):
        self.candidates = []
        self.best = None
        self.current_iteration = None
    
    def get_best(self):
        return self.best
    
    def update_candidates(self, candidate, current_iteration, **kwargs):
        if current_iteration > self.current_iteration:
            self.candidates = []
            self.current_iteration = current_iteration
        else:
            self.candidates.append(candidate)
            if self.best:
                if best.score < candidate.score:
                    self.best["bestN_last"] = candidate


class MapElitesCandidates(Candidates):
    def __init__(self, num_map_elites, **kwargs):
        self.candidates = {}
        self.best = {}
        self.best_all = {}
        self.best_scores = []
        self.num_map_elites = num_map_elites
    
    def get_best(self): 
        """Finds the top map elites"""
        self.best = {}
        self.best_scores = sorted(self.best_scores, key=lambda x:x[1]).reverse()

        num_actual_elites = min(self.num_map_elites, len(self.best_scores))
        for jdx in range(num_actual_elites):
            best_key = self.best_scores[jdx]
            self.best[best_key] = self.best_all[best_key]
        
        return self.best

    def _insert_into_best_scores(self,existing_key, new_score):
        for idx, (key,val) in enumerate(self.best_scores):
            if key==existing_key:
                self.best_scores.pop(idx)
                break
        self.best_scores.append((existing_key,new_score))

    def update_candidates(self, candidate, **kwargs):
        """Updates map elites... based on 'other' key """
        new_key = candidate.other

        # this map elites already exists
        if new_key in self.candidates:
            self.candidates[new_key].append(candidate)
            if self.best_all[new_key].score < candidate.score:
                self.best_all[new_key] = candidate
                self._insert_into_best_scores(new_key,candidate,score)
        
        # this is a new cluster
        else:
            self.candidates[new_key]=[candidate]
            self.best_all[new_key] = candidate
            self.best_scores.append((new_key, candidate.score))
        

