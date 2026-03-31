import cv2
import numpy as np

class Segmenter:
    def __init__(self):
        self.methods = {
            'otsu': self.otsu_threshold,
            'adaptive': self.adaptive_threshold,
            'canny': self.canny_edges,
            'watershed': self.watershed,
            'grabcut': self.grabcut
        }
    
    def otsu_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.astype(bool)
    
    def adaptive_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return mask.astype(bool)
    
    def canny_edges(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges.astype(bool)
    
    def watershed(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        
        markers = cv2.watershed(img, markers)
        mask = markers > 1
        return mask
    
    def grabcut(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h,w), np.uint8)
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)
        rect = (50, 50, w//2, h//2)
        
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype(bool)
        return mask2
    
    def compare_all_methods(self, img):
        results = {}
        for name, method in self.methods.items():
            mask = method(img)
            results[name] = mask
        return results
