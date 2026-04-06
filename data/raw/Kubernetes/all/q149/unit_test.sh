kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/demo-pod --timeout=60s

PLAYER_INITIAL_LIVES=$(kubectl get pod demo-pod -o=jsonpath='{.spec.containers[0].env[0].valueFrom.configMapKeyRef.name}')
UI_PROPERTIES_FILE_NAME=$(kubectl get pod demo-pod -o=jsonpath='{.spec.containers[0].env[1].valueFrom.configMapKeyRef.name}')
VOLUME=$(kubectl get pod demo-pod -o=jsonpath='{.spec.volumes[0].configMap.name}')

[ "$PLAYER_INITIAL_LIVES" = "game-demo" ] && \
[ "$UI_PROPERTIES_FILE_NAME" = "game-demo" ] && \
[ "$VOLUME" = "game-demo" ] && \
[ "$(kubectl exec demo-pod -- ls /config/game.properties | tr -d '\n')" = "/config/game.properties" ] && \
[ "$(kubectl exec demo-pod -- ls /config/user-interface.properties | tr -d '\n')" = "/config/user-interface.properties" ] && \
[ "$(kubectl exec demo-pod -- sh -c 'echo $PLAYER_INITIAL_LIVES')" = '3' ] && \
[ "$(kubectl exec demo-pod -- sh -c 'echo $UI_PROPERTIES_FILE_NAME')" = 'user-interface.properties' ] && \
echo cloudeval_unit_test_passed
