(defun c:C5 (/ a b c i)
	(setq a (getpoint "\n 중심점을 찍어주세요. : "))
	(setq b (getreal "\n 원의 지름을 입력하세요. : "))
	(setq c (/ b 2))
	(setq i 1)
	(repeat 5
		(command "circle" a (* c i))
		(setq i (+ i 1))
	)
	
	(princ "원 5개가 생성되었습니다.")
	(princ)
)

(defun c:lm ()
	(command "layer" "m" "중심선" "c" "1" "" "l" "center" ""
					 "m" "윤곽선" "c" "4" "" "l" "continuous" ""
					 "m" "외형선" "c" "3" "" "l" "continuous" ""
					 "m" "숨은선" "c" "2" "" "l" "hidden" ""
					 "m" "가상선" "c" "6" "" "l" "phantom" ""
					 "m" "구성선" "c" "8" "" "l" "continuous" ""
					 "m" "보조선" "c" "1" "" "l" "continuous" "" ""
	)

)

(defun c:1 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "중심선" "")
	)
	(setvar "clayer" "중심선")
	(princ)
)
(defun c:2 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "숨은선" "")
	)
	(setvar "clayer" "숨은선")
	(princ)
)
(defun c:3 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "외형선" "")
	)
	(setvar "clayer" "외형선")
	(princ)
)
(defun c:4 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "윤곽선" "")
	)
	(setvar "clayer" "윤곽선")
	(princ)
)
(defun c:5 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "보조선" "")
	)
	(setvar "clayer" "보조선")
	(princ)
)
(defun c:6 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "가상선" "")
	)
	(setvar "clayer" "가상선")
	(princ)
)
(defun c:7 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "0" "")
	)
	(setvar "clayer" "0")
	(princ)
)
(defun c:8 (/ ss)
	(setq ss (ssget))
	(if ss 
		(command "chprop" ss "" "la" "구성선" "")
	)
	(setvar "clayer" "구성선")
	(princ)
)
