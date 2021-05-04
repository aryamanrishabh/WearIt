import cv2
def bodyDetection(frame, face_cascade, filename):
    #video processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #Detect upper body
    face = face_cascade.detectMultiScale(frame_gray)
    alpha = 0
    
    # image manager
    image = cv2.imread(filename)
    rows,cols,channels = image.shape
    for (x,y,w,h) in face:
        #Calculate the upper part body coordinate
        body_start_x = x - int(1.0 * w)
        body_start_y = y + int(1.5 * h)
        body_end_x = int(2.0 * w) + x
        body_end_y = int(5.5 * h) + y
        width = int(2.0*w) + int(1.0*w)
        height = int(5.5*h) - int(1.5*h)
            #Cut the video image(roi)
        roi_gray = frame_gray[body_start_y : body_end_y, body_start_x : body_end_x]
        roi_color = frame[body_start_y : body_end_y, body_start_x : body_end_x]
            #Resize the image of clothes (The size of roi need to same with the image)
        image = cv2.resize(image,(width,height))
        roi_height = roi_color.shape[0]
        roi_width = roi_color.shape[1]
        img_height = image.shape[0]
        img_width = image.shape[1]
            #When the size of roi out of frame boundary, do not combine both roi and clothes image
            #as it will produce error
        if img_height != roi_height or img_width!=roi_width :
            cv2.imshow("My window", frame)
            return
        #When the size of roi not out of frame boundary, execute following code
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Create a mask of gray image and create its inverse mask also
        ret, mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
            # Black-out the area of clothes in ROI
        img_bg = cv2.bitwise_and(roi_color,roi_color,mask = mask_inv)
            # Take only region of clothes from clothes image.
        img_fg = cv2.bitwise_and(image,image,mask = mask)
            # Combine both foreground and background image
        img_fg = cv2.add(img_bg,img_fg)
           # Resize the image to fit on upper body size
        img_fg = cv2.resize(img_fg,(width,height))
            # Take the upper body frame region and combine with the img_fg
        added_image =  cv2.addWeighted(frame[ 0 :height, 0:width,:],alpha,img_fg[0 :height, 0: width,:],1-alpha,0)
            # Paste the combined image on the video frame
        frame[body_start_y:body_end_y, body_start_x:body_end_x] = added_image
    cv2.imshow("My window", frame)
