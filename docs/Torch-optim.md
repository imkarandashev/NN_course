# Torch optim  

`torch.optim` это пакет реализующий различные алгоритмы оптимизации. Самые популярные методы уже реализованы, и интерфейс достаточно общий, что бы в будущем легко интегрировать более сложные методы.

Чтобы использовать [`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) вы создать объект оптимизатора, который будет содержать состояния и обновлять параметры вычисления градиента.

Для создания [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) вы должны задать итерацию содержащую параметры оптимизации. Затем вы должны указать специфические для оптимизатора параметры, такие как скорость обучения.

Если вы хотите модель использующую GPU через `.cuda()`, то определите её до того как создадите оптимизатор. Параметры модели после `.cuda()` будут отличаться от тех, что были при создании.

Пример инициализации

```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Все оптимизаторы выполняются методом `step()`. Он обновляет параметры и вызывается как: `optimizer.step()`

Метод должен быть вызван после вычисления градиентов с помощью `backward()`.

Пример:

```
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

* * *

class `torch.optim.Optimizer`(_params_, _defaults_)

Базовый класс для всех оптимизаторов

*   params (_iterable_) – an iterable of [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) s or [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) s. Specifies what Tensors should be optimized.
*   defaults – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).
*   params (_iterable_) - итерация [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) или [`dict`](https://docs.python.org/3/library/stdtypes.html#dict). Определяет какие тензоры необходимо оптимизировать.
*   defaults – (dict): словарь содержащий значения по умолчанию для параметра оптимизатора.

Методы

`add_param_group`(_param_group_)

Добавляет параметры группы в [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) s param_groups.

Может быть полезно при точной настройке предварительно обученной сети, поскольку замороженный слой можно сделать обучаемым и добавить [`Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) по мере продвижения обучения.

`load_state_dict`(_state_dict_)

Загружает состояние оптимизатора

state_dict ([_dict_](https://docs.python.org/3/library/stdtypes.html#dict)) – состояние оптимизатора. Должен быть объектом который вернул [`state_dict()`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.state_dict).

`state_dict`()

Возвращает состояние оптимизатора как dict.

`step`(_closure_)

Выполняет один шаг оптимизатора(обновляет параметры).

сlosure (_callable_) – Замыкание, которое переоценивает модель и возвращает убыток. Необязательный для большинства оптимизаторов.

`zero_grad`(_set_to_none: bool = False_)

Устанавливает градиенты всех оптимизированных torch.Tensor на ноль.

set_to_none ([_bool_](https://docs.python.org/3/library/functions.html#bool)) – вместо нулевого значения устанавливает градиенты на None. Меняет некоторое поведение.

> **Алгоритмы оптимизации**

> `torch.optim.Adadelta`(_params_, _lr=1.0_, _rho=0.9_, _eps=1e-06_, _weight_decay=0_)

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   rho ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – коэффициент используемый для вычисления среднего квадрата градиента(по умолчанию: 0.9)
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию: 1e-6)
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – коэффициент который масштабирует дельта до её применения к параметрам (по умолчанию: 1.0)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)

> `torch.optim.Adagrad`(_params_, _lr=0.01_, _lr_decay=0_, _weight_decay=0_, _initial_accumulator_value=0_, _eps=1e-10_)

Реализует алгоритм Adagrad.

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-2)
*   lr_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – снижение скорости обучения (по умолчанию: 0)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию: 1e-10)

> `torch.optim.Adam`(_params_, _lr=0.001_, _betas=(0.9_, _0.999)_, _eps=1e-08_, _weight_decay=0_, _amsgrad=False_)

Реализует алгоритм Adam

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-3)
*   betas (_Tuple[_[_float_](https://docs.python.org/3/library/functions.html#float)_,_ [_float_](https://docs.python.org/3/library/functions.html#float)_], optional_) – коэффициенты, используемые для вычисления средних значений градиента и его квадрата (по умолчанию: (0.9, 0.999))
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию 1e-8)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)
*   amsgrad (_boolean, optional_) – другая версия алгоритма описанная в [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (default: False)

> `torch.optim.AdamW`(_params_, _lr=0.001_, _betas=(0.9_, _0.999)_, _eps=1e-08_, _weight_decay=0.01_, _amsgrad=False_)

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-3)
*   betas (_Tuple[_[_float_](https://docs.python.org/3/library/functions.html#float)_,_ [_float_](https://docs.python.org/3/library/functions.html#float)_], optional_) – коэффициенты, используемые для вычисления средних значений градиента и его квадрата (по умолчанию: (0.9, 0.999))
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию 1e-8)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)
*   amsgrad (_boolean, optional_) – следует ли использовать вариант этого алгоритма AMSGrad из статьи [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (default: False)

> `torch.optim.SparseAdam`(_params_, _lr=0.001_, _betas=(0.9_, _0.999)_, _eps=1e-08_)

*   Реализует ленивую версию алгоритма Adam, подходящую для разреженных тензоров.params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-3)
*   betas (_Tuple[_[_float_](https://docs.python.org/3/library/functions.html#float)_,_ [_float_](https://docs.python.org/3/library/functions.html#float)_], optional_) – коэффициенты, используемые для вычисления средних значений градиента и его квадрата (по умолчанию: (0.9, 0.999))
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию 1e-8)

> `torch.optim.Adamax`(_params_, _lr=0.002_, _betas=(0.9_, _0.999)_, _eps=1e-08_, _weight_decay=0_)

Реализует алгоритм Adamax (вариант Adam, основанный на норме бесконечности)

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 2e-3)
*   betas (_Tuple[_[_float_](https://docs.python.org/3/library/functions.html#float)_,_ [_float_](https://docs.python.org/3/library/functions.html#float)_], optional_) – коэффициенты, используемые для вычисления средних значений градиента и его квадрата
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию 1e-8)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)

> `torch.optim.ASGD`(_params_, _lr=0.01_, _lambd=0.0001_, _alpha=0.75_, _t0=1000000.0_, _weight_decay=0_)

Реализует усредненный стохастический градиентный спуск.

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-2)
*   lambd ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшающий коэффициент (по умолчанию: 1e-4)
*   alpha ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – мощность для обновления eta (defпо умолчаниюault: 0.75)
*   t0 ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – точка, с которой начинается усреднение (по умолчанию: 1e6)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)

> `torch.optim.LBFGS`(_params_, _lr=1_, _max_iter=20_, _max_eval=None_, _tolerance_grad=1e-07_, _tolerance_change=1e-09_, _history_size=100_, _line_search_fn=None_)

Реализует алгоритм L-BFGS, в значительной степени вдохновленный minFunc.

Этот оптимизатор не поддерживает параметры и группы параметров для отдельных параметров (может быть только один).

Сейчас все параметры должны быть на одном устройстве. Это будет улучшено в будущем.

Это оптимизатор с очень интенсивным использованием памяти (он требует дополнительных `param_bytes * (history_size + 1)` байтов ). Если он не помещается в памяти, попробуйте уменьшить размер истории или используйте другой алгоритм.

*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)) – скорость обучения (по умолчанию: 1)
*   max_iter ([_int_](https://docs.python.org/3/library/functions.html#int)) – максимальное количество итераций на шаг оптимизации (по умолчанию: 20)
*   max_eval ([_int_](https://docs.python.org/3/library/functions.html#int)) – максимальное количество вычислений функции за шаг оптимизации (по умолчанию: max_iter * 1.25).
*   tolerance_grad ([_float_](https://docs.python.org/3/library/functions.html#float)) – допуск на прерывание по оптимальности первого порядка (по умолчанию: 1e-5).
*   tolerance_change ([_float_](https://docs.python.org/3/library/functions.html#float)) – допуск прекращения при изменении значения функции / параметра (по умолчанию: 1e-9).
*   history_size ([_int_](https://docs.python.org/3/library/functions.html#int)) – размер истории обновлений (по умолчанию: 100).
*   line_search_fn ([_str_](https://docs.python.org/3/library/stdtypes.html#str)) – либо «strong_wolfe», либо None(по умолчанию: None).

> `torch.optim.RMSprop`(_params_, _lr=0.01_, _alpha=0.99_, _eps=1e-08_, _weight_decay=0_, _momentum=0_, _centered=False_)

Реализует RMSprop алгоритм.

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-2)
*   momentum ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – momentum factor (по умолчанию: 0)
*   alpha ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – smoothing constant (по умолчанию: 0.99)
*   eps ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – значение добавленное к знаменателю для повышение численной стабильности (по умолчанию 1e-8)
*   centered ([_bool_](https://docs.python.org/3/library/functions.html#bool)_, optional_) – если `True`, вычислить центрированный RMSProp, градиент нормализуется путем оценки его дисперсии
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)

`torch.optim.Rprop`(_params_, _lr=0.01_, _etas=(0.5_, _1.2)_, _step_sizes=(1e-06_, _50)_)

Реализует устойчивый алгоритм обратного распространения ошибки.

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – скорость обучения (по умолчанию: 1e-2)
*   etas (_Tuple[_[_float_](https://docs.python.org/3/library/functions.html#float)_,_ [_float_](https://docs.python.org/3/library/functions.html#float)_], optional_) –пара (etaminus, etaplis), которые являются мультипликативными факторами увеличения и уменьшения (по умолчанию: (0.5, 1.2))
*   step_sizes (_Tuple[_[_float_](https://docs.python.org/3/library/functions.html#float)_,_ [_float_](https://docs.python.org/3/library/functions.html#float)_], optional_) – пара минимальных и максимальных разрешенных размеров шага (по умолчанию: (1e-6, 50))

> `torch.optim.SGD`(_params_, _lr=<required parameter>_, _momentum=0_, _dampening=0_, _weight_decay=0_, _nesterov=False_)

Реализует стохастический градиентный спуск (опционально с импульсом).

*   params (_iterable_) – итерируемые параметры оптимизатора или словарь параметров по умолчанию.
*   lr ([_float_](https://docs.python.org/3/library/functions.html#float)) – скорость обучения
*   momentum ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – фактор импульса (по умолчанию: 0)
*   weight_decay ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – уменьшение веса (L2 штраф) (по умолчанию: 0)
*   dampening ([_float_](https://docs.python.org/3/library/functions.html#float)_, optional_) – гашение импульса (по умолчанию: 0)
*   nesterov ([_bool_](https://docs.python.org/3/library/functions.html#bool)_, optional_) – дает импульс Нестерова(по умолчанию: False)

Примеры:

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
```