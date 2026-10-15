(vl-load-com)

(defun tlen-get-length (ent)
  (vl-catch-all-apply
    '(lambda (obj)
       (vlax-curve-getDistAtParam
         obj
         (vlax-curve-getEndParam obj)
       )
     )
    (list ent)
  )
)

(defun c:TLEN (/ ss i ent lengthResult total count message)
  (prompt
    "\n전체 길이를 계산할 객체를 선택하세요: "
  )

  (setq ss
    (ssget
      '(
        (0 . "LINE,ARC,LWPOLYLINE,POLYLINE,SPLINE,CIRCLE,ELLIPSE")
       )
    )
  )

  (if ss
    (progn
      (setq i 0)
      (setq total 0.0)
      (setq count 0)

      (repeat (sslength ss)
        (setq ent (ssname ss i))
        (setq lengthResult (tlen-get-length ent))

        (if
          (not
            (vl-catch-all-error-p lengthResult)
          )
          (progn
            (setq total (+ total lengthResult))
            (setq count (1+ count))
          )
        )

        (setq i (1+ i))
      )

      (setq message
        (strcat
          "계산 객체 수: "
          (itoa count)
          "\n전체 길이: "
          (rtos total 2 2)
          " (도면 단위)"
        )
      )

      (alert message)
      (princ (strcat "\n" message))
    )
    (prompt "\n객체가 선택되지 않았습니다.")
  )

  (princ)
)

(princ "\nTLEN 명령이 로드되었습니다.")
(princ)