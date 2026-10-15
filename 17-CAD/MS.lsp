(defun mass2d-create-dcl (/ dclFile fp)
  (setq dclFile
    (vl-filename-mktemp "mass2d" nil ".dcl")
  )

  (setq fp (open dclFile "w"))

  (foreach line
    '(
      "mass2d : dialog {"
      "  label = \"질량 계산\";"
      "  : boxed_column {"
      "    label = \"재료 정보\";"
      "    : edit_box {"
      "      key = \"thickness\";"
      "      label = \"두께 (mm):\";"
      "      edit_width = 15;"
      "    }"
      "    : edit_box {"
      "      key = \"density\";"
      "      label = \"밀도 (g/cm3):\";"
      "      edit_width = 15;"
      "    }"
      "    : text {"
      "      key = \"error\";"
      "      label = \"\";"
      "      width = 35;"
      "    }"
      "  }"
      "  spacer;"
      "  : row {"
      "    alignment = centered;"
      "    : button {"
      "      key = \"accept\";"
      "      label = \"계산\";"
      "      is_default = true;"
      "      fixed_width = true;"
      "      width = 12;"
      "    }"
      "    : button {"
      "      key = \"cancel\";"
      "      label = \"취소\";"
      "      is_cancel = true;"
      "      fixed_width = true;"
      "      width = 12;"
      "    }"
      "  }"
      "}"
    )
    (write-line line fp)
  )

  (close fp)
  dclFile
)


(defun c:MS
  (/ ss dclFile dclId dialogResult
     thickness density totalArea
     validCount skipped i obj closedOk
     areaResult volume mass resultMessage)

  (vl-load-com)

  (prompt "\n질량을 계산할 닫힌 2D 객체를 선택하세요: ")
  (setq ss (ssget))

  (if ss
    (progn
      (setq dclFile (mass2d-create-dcl))
      (setq dclId (load_dialog dclFile))
      (setq dialogResult 0)

      (if
        (and
          (> dclId 0)
          (new_dialog "mass2d" dclId)
        )
        (progn
          (set_tile "thickness" "1.0")
          (set_tile "density" "7.85")

          (action_tile
            "accept"
            "(if
               (and
                 (> (atof (get_tile \"thickness\")) 0.0)
                 (> (atof (get_tile \"density\")) 0.0)
               )
               (progn
                 (setq thickness
                   (atof (get_tile \"thickness\"))
                 )
                 (setq density
                   (atof (get_tile \"density\"))
                 )
                 (done_dialog 1)
               )
               (set_tile
                 \"error\"
                 \"0보다 큰 숫자를 입력하세요.\"
               )
             )"
          )

          (action_tile
            "cancel"
            "(done_dialog 0)"
          )

          (setq dialogResult (start_dialog))
        )
        (alert "입력창을 불러오지 못했습니다.")
      )

      (if (> dclId 0)
        (unload_dialog dclId)
      )

      (if dclFile
        (vl-file-delete dclFile)
      )

      (if (= dialogResult 1)
        (progn
          (setq totalArea 0.0)
          (setq validCount 0)
          (setq skipped 0)
          (setq i 0)

          (repeat (sslength ss)
            (setq obj
              (vlax-ename->vla-object
                (ssname ss i)
              )
            )

            (setq closedOk
              (or
                (not
                  (vlax-property-available-p obj 'Closed)
                )
                (= (vla-get-Closed obj) :vlax-true)
              )
            )

            (if
              (and
                closedOk
                (vlax-property-available-p obj 'Area)
              )
              (progn
                (setq areaResult
                  (vl-catch-all-apply
                    'vla-get-Area
                    (list obj)
                  )
                )

                (if
                  (vl-catch-all-error-p areaResult)
                  (setq skipped (1+ skipped))
                  (progn
                    (setq totalArea
                      (+ totalArea areaResult)
                    )
                    (setq validCount (1+ validCount))
                  )
                )
              )
              (setq skipped (1+ skipped))
            )

            (setq i (1+ i))
          )

          (if (> validCount 0)
            (progn
              (setq volume
                (* totalArea thickness)
              )

              (setq mass
                (* (/ volume 1000.0) density)
              )

              (setq resultMessage
                (strcat
                  "계산 객체 수: "
                  (itoa validCount)
                  "\n전체 면적: "
                  (rtos totalArea 2 2)
                  " mm2"
                  "\n두께: "
                  (rtos thickness 2 2)
                  " mm"
                  "\n밀도: "
                  (rtos density 2 3)
                  " g/cm3"
                  "\n질량: "
                  (rtos mass 2 2)
                  " g"
                )
              )

              (alert resultMessage)
              (princ (strcat "\n" resultMessage))

              (if (> skipped 0)
                (princ
                  (strcat
                    "\n제외된 객체 수: "
                    (itoa skipped)
                  )
                )
              )
            )
            (alert
              "계산 가능한 닫힌 2D 객체가 없습니다."
            )
          )
        )
      )
    )
    (prompt "\n객체가 선택되지 않았습니다.")
  )

  (princ)
)