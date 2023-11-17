
    def recognizeOne(self,src):
        x_tempx = src
        x_temp = cv2.resize(x_tempx,( 164,48))
        x_temp = x_temp.transpose(1, 0, 2)
        y_pred = self.modelSeqRec.predict(np.array([x_temp]))
        y_pred = y_pred[:,2:,:]
        return self.fastdecode(y_pred)
