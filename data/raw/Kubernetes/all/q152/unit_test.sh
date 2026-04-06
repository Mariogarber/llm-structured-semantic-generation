kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/spread-pod --timeout=60s

[ "$(kubectl get pod spread-pod -o=jsonpath='{.spec.topologySpreadConstraints[0].maxSkew}')" = "2" ] && \
[ "$(kubectl get pod spread-pod -o=jsonpath='{.spec.topologySpreadConstraints[0].topologyKey}')" = "kubernetes.io/hostname" ] && \
[ "$(kubectl get pod spread-pod -o=jsonpath='{.spec.topologySpreadConstraints[0].whenUnsatisfiable}')" = "DoNotSchedule" ] && \
[ "$(kubectl get pod spread-pod -o=jsonpath='{.spec.topologySpreadConstraints[1].maxSkew}')" = "1" ] && \
[ "$(kubectl get pod spread-pod -o=jsonpath='{.spec.topologySpreadConstraints[1].topologyKey}')" = "topology.kubernetes.io/zone" ] && \
[ "$(kubectl get pod spread-pod -o=jsonpath='{.spec.topologySpreadConstraints[1].whenUnsatisfiable}')" = "ScheduleAnyway" ] && \
echo cloudeval_unit_test_passed
