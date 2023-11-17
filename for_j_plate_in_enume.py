
        for j,plate in enumerate(images):
            plate, rect  =plate
            image_rgb,rect_refine = self.finemappingVertical(plate,rect)
            res,confidence = self.recognizeOne(image_rgb)
            res_set.append([res,confidence,rect_refine])
