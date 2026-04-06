kubectl apply -f labeled_code.yaml
sleep 3

[ "$(kubectl get svc my-service -o jsonpath='{.spec.clusterIP}')" = "10.96.0.164" ] && \
[ "$(kubectl get svc my-service -o jsonpath='{.spec.ipFamilies[0]}')" = "IPv4" ] && \
[ "$(kubectl get svc my-service -o jsonpath='{.spec.ipFamilyPolicy}')" = "SingleStack" ] && \
[ "$(kubectl get svc my-service -o jsonpath='{.spec.ports[*].port}')" -eq 80 ] && \
[ "$(kubectl get svc my-service -o jsonpath='{.spec.ports[*].targetPort}')" -eq 9376 ] && \
[ "$(kubectl get svc my-service -o jsonpath='{.spec.sessionAffinity}')" = "None" ] && \
echo cloudeval_unit_test_passed
